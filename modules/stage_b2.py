# Code adapted from https://github.com/Stability-AI/StableCascade, https://github.com/comfyanonymous/ComfyUI/

"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from comfy.ldm.cascade.common import AttnBlock, LayerNorm2d_op, ResBlock, FeedForwardBlock, TimestepBlock

import copy
from einops import rearrange

from ..latents import tile_latent, untile_latent, gaussian_blur_2d, median_blur_2d
from ..style_transfer import apply_scattersort_masked, apply_scattersort_tiled, adain_seq_inplace, adain_patchwise_row_batch_med, adain_patchwise_row_batch, StyleCascadeC_Model, Retrojector

from ..modules.common_std import ReAttnBlock, ReAttention2D, ReOptimizedAttention

class StageB2(nn.Module):
    def __init__(self, c_in=4, c_out=4, c_r=64, patch_size=2, c_cond=1280, c_hidden=[320, 640, 1280, 1280],
                 nhead=[-1, -1, 20, 20], blocks=[[2, 6, 28, 6], [6, 28, 6, 2]],
                 block_repeat=[[1, 1, 1, 1], [3, 3, 2, 2]], level_config=['CT', 'CT', 'CTA', 'CTA'], c_clip=1280,
                 c_clip_seq=4, c_effnet=16, c_pixels=3, kernel_size=3, dropout=[0, 0, 0.0, 0.0], self_attn=True,
                 t_conds=['sca'], stable_cascade_stage=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        self.effnet_batch = None
        self.effnet_batch_maps = None
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)

        # CONDITIONING
        self.effnet_mapper = nn.Sequential(
            operations.Conv2d(c_effnet, c_hidden[0] * 4, kernel_size=1, dtype=dtype, device=device),
            nn.GELU(),
            operations.Conv2d(c_hidden[0] * 4, c_hidden[0], kernel_size=1, dtype=dtype, device=device),
            LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        )
        self.pixels_mapper = nn.Sequential(
            operations.Conv2d(c_pixels, c_hidden[0] * 4, kernel_size=1, dtype=dtype, device=device),
            nn.GELU(),
            operations.Conv2d(c_hidden[0] * 4, c_hidden[0], kernel_size=1, dtype=dtype, device=device),
            LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        )
        self.clip_mapper = operations.Linear(c_clip, c_cond * c_clip_seq, dtype=dtype, device=device)
        self.clip_norm = operations.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            operations.Conv2d(c_in * (patch_size ** 2), c_hidden[0], kernel_size=1, dtype=dtype, device=device),
            LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == 'C':
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'A':
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'F':
                return FeedForwardBlock(c_hidden, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'T':
                return TimestepBlock(c_hidden, c_r, conds=t_conds, dtype=dtype, device=device, operations=operations)
            else:
                raise Exception(f'Block type {block_type} not supported')

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(nn.Sequential(
                    LayerNorm2d_op(operations)(c_hidden[i - 1], elementwise_affine=False, eps=1e-6, dtype=dtype, device=device),
                    operations.Conv2d(c_hidden[i - 1], c_hidden[i], kernel_size=2, stride=2, dtype=dtype, device=device),
                ))
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i], self_attn=self_attn[i])
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(operations.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1, dtype=dtype, device=device))
                self.down_repeat_mappers.append(block_repeat_mappers)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(nn.Sequential(
                    LayerNorm2d_op(operations)(c_hidden[i], elementwise_affine=False, eps=1e-6, dtype=dtype, device=device),
                    operations.ConvTranspose2d(c_hidden[i], c_hidden[i - 1], kernel_size=2, stride=2, dtype=dtype, device=device),
                ))
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i],
                                      self_attn=self_attn[i])
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(operations.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1, dtype=dtype, device=device))
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-6, dtype=dtype, device=device),
            operations.Conv2d(c_hidden[0], c_out * (patch_size ** 2), kernel_size=1, dtype=dtype, device=device),
            nn.PixelShuffle(patch_size),
        )
        
        self.Retrojector = Retrojector(self.embedding[1], pinv_dtype=torch.float64, dtype=torch.float64)
        
        for down_block_stage in self.down_blocks:
            for down_block in down_block_stage:
                if isinstance(down_block, AttnBlock):
                    down_block.__class__ = ReAttnBlock
                    down_block.attention.__class__ = ReAttention2D
                    down_block.attention.attn.__class__ = ReOptimizedAttention
                    
        for up_block_stage in self.up_blocks:
            for up_block in up_block_stage:
                if isinstance(up_block, AttnBlock):
                    up_block.__class__ = ReAttnBlock
                    up_block.attention.__class__ = ReAttention2D
                    up_block.attention.attn.__class__ = ReOptimizedAttention
        

        # --- WEIGHT INIT ---
    #     self.apply(self._init_weights)  # General init
    #     nn.init.normal_(self.clip_mapper.weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.effnet_mapper[0].weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.effnet_mapper[2].weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.pixels_mapper[0].weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.pixels_mapper[2].weight, std=0.02)  # conditionings
    #     torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
    #     nn.init.constant_(self.clf[1].weight, 0)  # outputs
    # 
    #     # blocks
    #     for level_block in self.down_blocks + self.up_blocks:
    #         for block in level_block:
    #             if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
    #                 block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks[0]))
    #             elif isinstance(block, TimestepBlock):
    #                 for layer in block.modules():
    #                     if isinstance(layer, nn.Linear):
    #                         nn.init.constant_(layer.weight, 0)
    # 
    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def gen_c_embeddings(self, clip):
        if len(clip.shape) == 2:
            clip = clip.unsqueeze(1)
        clip = self.clip_mapper(clip).view(clip.size(0), clip.size(1) * self.c_clip_seq, -1)
        clip = self.clip_norm(clip)
        return clip

    def _down_encode(self, x, r_embed, clip, style_blocks=None):
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers, style_blocks)
        for down_block, downscaler, repmap, style_block in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, ResBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, ResBlock)):
                        x = block(x)
                        x = style_block(x, "res")
                    elif isinstance(block, AttnBlock) or isinstance(block, ReAttnBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, AttnBlock)):
                        x = block(x, clip, style_block.attn_block)
                        x = style_block(x, "attn")
                    elif isinstance(block, TimestepBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, TimestepBlock)):
                        x = block(x, r_embed)
                        x = style_block(x, "timestep")
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
        return level_outputs


    def sag_attn_proc(self, x, clip, block, sag_func, n_heads=24, dim_head=48):
        orig_x = x

        extra_options={}
        extra_options["n_heads"] = n_heads #self.n_heads 24 * 48 = 1152... q v k are b,seq_len,1152
        extra_options["dim_head"] = dim_head #self.d_head
        extra_options["attn_precision"] = None 
        extra_options["cond_or_uncond"] = [1,0]
        
        x = block.norm(x)
        kv = block.kv_mapper(clip)
        orig_shape = x.shape
        
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        kv = torch.cat([x, kv], dim=1) #previously conditional: if self_attn:
        q_ = block.attention.attn.to_q(x)
        k_ = block.attention.attn.to_k(kv)
        v_ = block.attention.attn.to_v(kv)
        
        x = sag_func(q_, k_, v_, extra_options)
        x = block.attention.attn.out_proj(x)
        x = orig_x + x.permute(0, 2, 1).view(*orig_shape)

        return x
    
    def pag_attn_proc(self, x, clip, block, rag_func=None):
        orig_x = x
        
        x = block.norm(x)
        #kv = block.kv_mapper(clip)
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4

        #kv = torch.cat([x, kv], dim=1) # unrolled attention blocks here
        #q_ = block.attention.attn.to_q(x)
        #k_ = block.attention.attn.to_k(kv)
        #v_ = block.attention.attn.to_v(kv)
        x_q = block.attention.attn.to_q(x)
        x_k = block.attention.attn.to_k(x)
        x_v = block.attention.attn.to_v(x)
        x   = block.attention.attn.out_proj(x_v)
        
        #x = sag_func(q_, k_, v_, extra_options)
        #x = block.attention.attn.out_proj(x)
        x = orig_x + x.permute(0, 2, 1).view(*orig_shape)        
        
        return x
    
    def rag_attn_proc(self, x, clip, block, rag_func=None):
        #x = torch.randn_like(x).to('cuda')

        #gaussian_blur = torchvision.transforms.GaussianBlur(3, sigma=2.0)
        #x = gaussian_blur(x)
        x = block(torch.randn_like(x).to('cuda'), clip) #random query
        #x = torch.randn_like(x).to('cuda')
        #x = block(torch.full_like(x, 0.0).to('cuda'), clip)
        
        return x

    def _up_decode(self, level_outputs, r_embed, clip, pag_patch_flag=False, sag_func=None, style_blocks=None):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers, style_blocks)
        for i, (up_block, upscaler, repmap, style_block) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, ResBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, ResBlock)):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                            x = torch.nn.functional.interpolate(x, skip.shape[-2:], mode='bilinear', align_corners=True)
                        x = block(x, skip)
                        x = style_block(x, "res")
                    elif isinstance(block, AttnBlock) or isinstance(block, ReAttnBlock) or (hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, AttnBlock)):
                        
                        if i == 0 and j ==0 and k == 2 and sag_func != None: 
                            x = self.sag_attn_proc(x, clip, block, sag_func)
                        elif i == 0 and j ==0 and k == 2 and pag_patch_flag == "rag" and sag_func == None:
                            x = self.rag_attn_proc(x, clip, block)
                        elif i == 0 and j ==0 and k == 2 and pag_patch_flag == "pag" and sag_func == None:
                            x = self.pag_attn_proc(x, clip, block)
                        else:
                            x = block(x, clip, style_block.attn_block)
                        x = style_block(x, "attn")
                    elif isinstance(block, TimestepBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module, TimestepBlock)):
                        x = block(x, r_embed)
                        x = style_block(x, "timestep")
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x
    
    def set_effnet_batch(self, effnet_batch=None):
        self.effnet_batch = effnet_batch
        
    def set_effnet_batch_maps(self, effnet_batch_maps=None):
        self.effnet_batch_maps = effnet_batch_maps


    def invert_unshuffle_conv(
        self,
        unshuffle: torch.nn.PixelUnshuffle,
        conv1x1:   torch.nn.Conv2d,
        y:         torch.Tensor,
        original_shape: torch.Size
    ) -> torch.Tensor:

        conv1x1 = copy.deepcopy(conv1x1).to(torch.float64)
        y = y.to(torch.float64)
        
        B, C_in, H, W = original_shape
        r = unshuffle.downscale_factor
        B2, C_out, Hr, Wr = y.shape
        assert B2 == B
        assert Hr == H//r and Wr == W//r

        b = conv1x1.bias.view(1, C_out, 1, 1)
        y_nobias = y - b

        W = conv1x1.weight.view(C_out, -1) 
        W_pinv = torch.linalg.pinv(W)  

        y_flat = y_nobias.reshape(B, C_out, -1) 

        x_unshuf_flat = W_pinv @ y_flat  
        x_unshuf = x_unshuf_flat.view(B, C_in*r*r, Hr, Wr)

        x_rec = F.pixel_shuffle(x_unshuf, upscale_factor=r) 
        return x_rec


    def forward(self, x, r, effnet, clip, pixels=None, **kwargs):

        sigmas = kwargs['transformer_options']['sigmas']
        transformer_options = kwargs['transformer_options']
        SIGMA = transformer_options['sigmas'] # timestep[0].unsqueeze(0) #/ 1000
        
        h_len, w_len = x.shape[-2:]
        img_len = h_len * w_len
        img_slice = slice(None, -1) #slice(None, img_len)   # for the sake of cross attn... :-1
        txt_slice = slice(None, -1)
        HEADS=0
        StyleMMDiT = transformer_options.get('StyleMMDiT', StyleCascadeC_Model())        
        StyleMMDiT.set_len(h_len, w_len, img_slice, txt_slice, HEADS=HEADS)
        StyleMMDiT.Retrojector = self.Retrojector if hasattr(self, "Retrojector") else None
        transformer_options['StyleMMDiT'] = None

        StyleMMDiT.Retrojector.unshuffle = self.embedding[0]
        StyleMMDiT.Retrojector.embedder = copy.deepcopy(self.embedding).to(torch.float64).cuda()
        #StyleMMDiT.Retrojector.embedder[1].weight.data = StyleMMDiT.Retrojector.embedder[1].weight.data.cuda()
        #StyleMMDiT.Retrojector.embedder[1].bias.data = StyleMMDiT.Retrojector.embedder[1].bias.data.cuda()

        x_orig = x.clone()
        
        x_tmp = transformer_options.get("x_tmp")
        if x_tmp is not None:
            x_tmp = x_tmp.clone() / ((SIGMA ** 2 + 1) ** 0.5)
            x_tmp = x_tmp.expand_as(x) # (x.shape[0], -1, -1, -1) # .clone().to(x)
        y0_style, img_y0_style = None, None
        
        z_ = transformer_options.get("z_")   # initial noise and/or image+noise from start of rk_sampler_beta() 
        rk_row = transformer_options.get("row") # for "smart noise"
        if z_ is not None:
            x_init = z_[rk_row].to(x)
        elif 'x_init' in transformer_options:
            x_init = transformer_options.get('x_init').to(x)
        
        if pixels is None:
            pixels = x.new_zeros(x.size(0), 3, 8, 8)



        
        x_orig, r_orig, clip_orig = clone_inputs(x, r, clip)
        new_x = None
        
        
        # recon loop to extract exact noise pred for scattersort guide assembly
        RECON_MODE = StyleMMDiT.noise_mode == "recon"
        recon_iterations = 2 if StyleMMDiT.noise_mode == "recon" else 1
        for recon_iter in range(recon_iterations):
            y0_style = StyleMMDiT.guides
            y0_style_active = True if type(y0_style) == torch.Tensor else False
            
            RECON_MODE = True     if StyleMMDiT.noise_mode == "recon" and recon_iter == 0     else False
            
            ISIGMA = SIGMA
            if StyleMMDiT.noise_mode == "recon" and recon_iter == 1: 
                ISIGMA = SIGMA #* EO("ISIGMA_FACTOR", 1.0)
                
                model_sampling = transformer_options.get('model_sampling')     
                r_orig = model_sampling.timestep(ISIGMA).expand_as(r_orig)
                
                x_recon = x_tmp if x_tmp is not None else x_orig
                #noise_prediction = x_recon + (1-SIGMA.to(x_recon)) * eps.to(x_recon)
                noise_prediction = eps.to(x_recon)
                denoised = x_recon * ((SIGMA.to(x_recon) ** 2 + 1) ** 0.5)   -   SIGMA.to(x_recon) * eps.to(x_recon)
                
                denoised = StyleMMDiT.apply_recon_lure(denoised, y0_style.to(x_recon))   # .to(denoised)

                new_x = (denoised + ISIGMA.to(x_recon) * noise_prediction) / ((ISIGMA.to(x_recon) ** 2 + 1) ** 0.5)
                #h_orig = new_x.clone().to(x)
                x_init = noise_prediction
            elif StyleMMDiT.noise_mode == "bonanza":
                x_init = torch.randn_like(x_init)

            if y0_style_active:
                if y0_style.sum() == 0.0 and y0_style.std() == 0.0:
                    y0_style_noised = x.clone()
                else:
                    y0_style_noised = (y0_style + ISIGMA.to(y0_style) * x_init.expand_as(x_orig).to(y0_style)) / ((ISIGMA.to(y0_style) ** 2 + 1) ** 0.5)    #x_init.expand(x.shape[0],-1,-1,-1).to(y0_style)) 

            out_list = []
            for cond_iter in range(len(transformer_options['cond_or_uncond'])):
                UNCOND = transformer_options['cond_or_uncond'][cond_iter] == 1
                
                bsz_style = y0_style.shape[0] if y0_style_active else 0
                bsz       = 1 if RECON_MODE else bsz_style + 1
                
                x, r, clip = clone_inputs(x_orig[cond_iter].unsqueeze(0), r_orig[cond_iter].unsqueeze(0), clip_orig[cond_iter].unsqueeze(0))
                x = new_x[cond_iter].unsqueeze(0).to(x) if new_x is not None else x
        
                if y0_style_active and not RECON_MODE:
                    clip = clip[0:1].repeat(bsz,1,1)
                    x = torch.cat([x, y0_style_noised[cond_iter:cond_iter+1]], dim=0).to(x)
                    """if mask is None:
                        context, y, _ = StyleMMDiT.apply_style_conditioning(
                            UNCOND       = UNCOND,
                            base_context = context,
                            base_y       = y,
                            base_llama3  = None,
                        )
                    else:
                        context = context.repeat(bsz_style + 1, 1, 1)
                        y = y.repeat(bsz_style + 1, 1)                   if y      is not None else None
                    h = torch.cat([h, y0_style_noised[cond_iter:cond_iter+1]], dim=0).to(h)"""
                    


                # Model Blocks
                x = self.embedding(x) # 1,4,240,400 -> 1,320,120,200 # patchified?

                # Process the conditioning embeddings
                r_embed = self.gen_r_embedding(r).to(dtype=x.dtype)
                for c in self.t_conds:
                    t_cond = kwargs.get(c, torch.zeros_like(r))
                    t_cond = t_cond[:effnet.shape[0]]
                    
                    r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond).to(dtype=x.dtype)], dim=1)
                clip = self.gen_c_embeddings(clip) # 1,1280 -> 1,4,1280   and normalized

                if self.effnet_batch_maps is None:
                    if hasattr(StyleMMDiT, "latent_effnet"):
                        effnet = torch.cat([effnet, StyleMMDiT.latent_effnet.to(effnet)], dim=0)
                    effnet_mapper_output = self.effnet_mapper( nn.functional.interpolate(effnet, size=x.shape[-2:], mode='bilinear', align_corners=True))
                    effnet_mapper_output = StyleMMDiT(effnet_mapper_output, "proj_in")
                    effnet_mapper_output = effnet_mapper_output[0:x.shape[0]] if x.shape[0] < effnet_mapper_output.shape[0] else effnet_mapper_output
                    x = x + effnet_mapper_output
                    x = StyleMMDiT(x, "proj_in")
                else: 
                    effnet_mapper_output = self.effnet_mapper( nn.functional.interpolate(effnet, size=x.shape[-2:], mode='bilinear', align_corners=True))
                    x = x + self.effnet_batch_maps.repeat(x.shape[0], *([1]*(x.ndim-1)))
                    
                x = x + nn.functional.interpolate(self.pixels_mapper(pixels), size=x.shape[-2:], mode='bilinear', align_corners=True)
                level_outputs = self._down_encode(x, r_embed, clip, style_blocks=StyleMMDiT.input_blocks)
                
                pag_patch_flag=""
                if 'patches_replace' in kwargs['transformer_options']:
                    if "attn1" in kwargs['transformer_options']['patches_replace']:
                        pag_patch_flag = "rag"
                    if "attn1_pag" in kwargs['transformer_options']['patches_replace']:
                        pag_patch_flag = "pag"
                
                # pag_patch_flag = True if 'patches_replace' in kwargs['transformer_options'] else False
                sag_func = (kwargs.get('transformer_options', {}).get('patches_replace', {}).get('attn1', {}).get(('middle', 0, 0), None))
                x = self._up_decode(level_outputs, r_embed, clip, pag_patch_flag=pag_patch_flag, sag_func=sag_func, style_blocks=StyleMMDiT.output_blocks)
                
                eps = self.clf(x)
                x = StyleMMDiT(x, "proj_out")

                out_list.append(eps[0:1])
                
            eps = torch.stack(out_list, dim=0).squeeze(dim=1)

            if recon_iter == 1:
                denoised = new_x * ((ISIGMA ** 2 + 1) ** 0.5)  - ISIGMA.to(new_x) * eps.to(new_x)
                if x_tmp is not None:
                    eps = (x_tmp * ((SIGMA ** 2 + 1) ** 0.5) - denoised.to(x_tmp)) / SIGMA.to(x_tmp)
                else:
                    eps = (x_orig * ((SIGMA ** 2 + 1) ** 0.5) - denoised.to(x_orig)) / SIGMA.to(x_orig)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        y0_style_pos        = transformer_options.get("y0_style_pos")
        y0_style_neg        = transformer_options.get("y0_style_neg")

        y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight",    0.0)
        y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight", 0.0)
        y0_style_pos_synweight *= y0_style_pos_weight

        y0_style_neg_weight    = transformer_options.get("y0_style_neg_weight",    0.0)
        y0_style_neg_synweight = transformer_options.get("y0_style_neg_synweight", 0.0)
        y0_style_neg_synweight *= y0_style_neg_weight
        
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        pinv_dtype = torch.float32 if dtype != torch.float64 else dtype
        W_inv = None
        
        if eps.shape[0] == 2 or (eps.shape[0] == 1): #: and not UNCOND):
            if y0_style_pos is not None and y0_style_pos_weight != 0.0:
                y0_style_pos = y0_style_pos.to(torch.float64)
                x   = x_orig.clone().to(torch.float64) * ((SIGMA ** 2 + 1) ** 0.5)
                eps = eps.to(torch.float64)
                eps_orig = eps.clone()
            
                sigma = SIGMA
                denoised = x - sigma * eps

                pixel_unshuffler = self.embedding[0]
                x_embedder = copy.deepcopy(self.embedding[1]).to(denoised)
                
                denoised_embed = x_embedder(pixel_unshuffler(denoised))
                y0_adain_embed = x_embedder(pixel_unshuffler(y0_style_pos))

                denoised_embed = rearrange(denoised_embed, "B C H W -> B (H W) C")
                y0_adain_embed = rearrange(y0_adain_embed, "B C H W -> B (H W) C")

                if transformer_options['y0_style_method'] == "AdaIN":
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    """for adain_iter in range(EO("style_iter", 0)):
                        denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                        denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                        denoised_embed = F.linear(denoised_embed.to(W), W, b).to(img)
                        denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)"""
                        
                elif transformer_options['y0_style_method'] == "WCT":
                    if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                        self.y0_adain_embed = y0_adain_embed
                        
                        f_s          = y0_adain_embed[0].clone()
                        self.mu_s    = f_s.mean(dim=0, keepdim=True)
                        f_s_centered = f_s - self.mu_s
                        
                        cov = (f_s_centered.transpose(-2,-1).double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

                        S_eig, U_eig = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        S_eig_sqrt    = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
                        
                        whiten = U_eig @ torch.diag(S_eig_sqrt) @ U_eig.T
                        self.y0_color  = whiten.to(f_s_centered)

                    for wct_i in range(eps.shape[0]):
                        f_c          = denoised_embed[wct_i].clone()
                        mu_c         = f_c.mean(dim=0, keepdim=True)
                        f_c_centered = f_c - mu_c
                        
                        cov = (f_c_centered.transpose(-2,-1).double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                        S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                        
                        whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                        whiten = whiten.to(f_c_centered)

                        f_c_whitened = f_c_centered @ whiten.T
                        f_cs         = f_c_whitened @ self.y0_color.T.to(f_c_whitened) + self.mu_s.to(f_c_whitened)
                        
                        denoised_embed[wct_i] = f_cs

                
                denoised_embed = rearrange(denoised_embed, "B (H W) C -> B C H W", W=eps.shape[-1] // 2)
                denoised_approx = self.invert_unshuffle_conv(pixel_unshuffler, x_embedder, denoised_embed, x_orig.shape)
                denoised_approx = denoised_approx.to(eps)

                
                eps = (x - denoised_approx) / sigma
                
                #UNCOND = transformer_options['cond_or_uncond'][cond_iter] == 1

                if eps.shape[0] == 1 and transformer_options['cond_or_uncond'][0] == 1:
                    eps[0] = eps_orig[0] + y0_style_neg_synweight * (eps[0] - eps_orig[0])
                    #if eps.shape[0] == 2:
                    #    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
                else: #if not UNCOND:
                    if eps.shape[0] == 2:
                        eps[1] = eps_orig[1] + y0_style_pos_weight * (eps[1] - eps_orig[1])
                        eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                    else:
                        eps[0] = eps_orig[0] + y0_style_pos_weight * (eps[0] - eps_orig[0])
                
                eps = eps.float()
        
        if eps.shape[0] == 2 or (eps.shape[0] == 1): # and UNCOND):
            if y0_style_neg is not None and y0_style_neg_weight != 0.0:
                y0_style_neg = y0_style_neg.to(torch.float64)
                x   = x_orig.clone().to(torch.float64)* ((SIGMA ** 2 + 1) ** 0.5)
                eps = eps.to(torch.float64)
                eps_orig = eps.clone()
                
                sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
                denoised = x - sigma * eps

                pixel_unshuffler = self.embedding[0]
                x_embedder = copy.deepcopy(self.embedding[1]).to(denoised)
                
                denoised_embed = x_embedder(pixel_unshuffler(denoised))
                y0_adain_embed = x_embedder(pixel_unshuffler(y0_style_neg))

                denoised_embed = rearrange(denoised_embed, "B C H W -> B (H W) C")
                y0_adain_embed = rearrange(y0_adain_embed, "B C H W -> B (H W) C")

                if transformer_options['y0_style_method'] == "AdaIN":
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    """for adain_iter in range(EO("style_iter", 0)):
                        denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                        denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                        denoised_embed = F.linear(denoised_embed.to(W), W, b).to(img)
                        denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)"""
                        
                elif transformer_options['y0_style_method'] == "WCT":
                    if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                        self.y0_adain_embed = y0_adain_embed
                        
                        f_s          = y0_adain_embed[0].clone()
                        self.mu_s    = f_s.mean(dim=0, keepdim=True)
                        f_s_centered = f_s - self.mu_s
                        
                        #cov = (f_s_centered.T.double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)
                        cov = (f_s_centered.transpose(-2,-1).double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

                        S_eig, U_eig = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        S_eig_sqrt    = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
                        
                        whiten = U_eig @ torch.diag(S_eig_sqrt) @ U_eig.T
                        self.y0_color  = whiten.to(f_s_centered)

                    for wct_i in range(eps.shape[0]):
                        f_c          = denoised_embed[wct_i].clone()
                        mu_c         = f_c.mean(dim=0, keepdim=True)
                        f_c_centered = f_c - mu_c
                        
                        #cov = (f_c_centered.T.double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)
                        cov = (f_c_centered.transpose(-2,-1).double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                        S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                        
                        whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                        whiten = whiten.to(f_c_centered)

                        f_c_whitened = f_c_centered @ whiten.T
                        f_cs         = f_c_whitened @ self.y0_color.T.to(f_c_whitened) + self.mu_s.to(f_c_whitened)
                        
                        denoised_embed[wct_i] = f_cs

                denoised_embed = rearrange(denoised_embed, "B (H W) C -> B C H W", W=eps.shape[-1] // 2)
                denoised_approx = self.invert_unshuffle_conv(pixel_unshuffler, x_embedder, denoised_embed, x_orig.shape)
                denoised_approx = denoised_approx.to(eps)
                
                
                if eps.shape[0] == 1 and not transformer_options['cond_or_uncond'][0] == 1:
                    eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                else:
                    eps = (x - denoised_approx) / sigma
                    eps[0] = eps_orig[0] + y0_style_neg_weight * (eps[0] - eps_orig[0])
                    if eps.shape[0] == 2:
                        eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
                
            eps = eps.float()
        
        return eps





    

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data.clone().to(self_params.device) * (1 - beta)
        for self_buffers, src_buffers in zip(self.buffers(), src_model.buffers()):
            self_buffers.data = self_buffers.data * beta + src_buffers.data.clone().to(self_buffers.device) * (1 - beta)


    
def clone_inputs(*args, index: int = None):
    if index is None:
        return tuple(x.clone() if x is not None else None for x in args)
    else:
        return tuple(x[index].unsqueeze(0).clone() if x is not None else None for x in args)

