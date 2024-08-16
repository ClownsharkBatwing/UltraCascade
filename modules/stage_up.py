import torch
from torch import nn
import torchvision
import functools

from comfy.samplers import CFGGuider
from comfy.ldm.cascade.common import AttnBlock, ResBlock, TimestepBlock
from ..modules.inr_fea_res_lite import TransInr, ScaleNormalize_res
from comfy.ldm.cascade.stage_c import StageC

def ultracascade_reset_hook(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        UltraCascadePatch.reset_sampling()
        return func(self, *args, **kwargs)
    return wrapper

original_sample = CFGGuider.sample
CFGGuider.sample = ultracascade_reset_hook(original_sample)

class UltraCascadePatch:
    current_step = 0
    x_lr = None
    guide_weights = None
    guide_type = 'residual'

    @classmethod
    def reset_sampling(cls):
        cls.current_step = 0
    def __init__(self, x_lr, guide_weights, guide_type):
        self.x_lr = x_lr
        self.guide_weights = guide_weights
        self.guide_type = guide_type
        self.lr_guide = None

    def __call__(self, x, r, clip_text, clip_text_pooled, clip_img, extra_options):
        if self.x_lr is None:
            return x
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if self.lr_guide is None or self.__class__.current_step == 0:
                self.lr_guide = None
                self.__class__.current_step = 0
                self.lr_guide = self.generate_lr_guide(x, r, clip_text, clip_text_pooled, clip_img, extra_options)

            if self.guide_weights is not None:
                guide_weight = self.guide_weights[self.__class__.current_step]
            else:
                guide_weight = r[0].item()

        self.__class__.current_step += 1
        
        x = self.apply_guide(x, self.lr_guide, guide_weight, extra_options)

        return x
    
    def update(self, x_lr, guide_weights, guide_type, reset=True):
        if x_lr is not None:
            self.x_lr = x_lr
        if guide_weights is not None:
            self.guide_weights = guide_weights
        if guide_type is not None:
            self.guide_type = guide_type
        
        if reset:
            self.lr_guide = None
            self.current_step = 0

    def generate_lr_guide(self, x, r, clip_text, clip_text_pooled, clip_img, extra_options):
        r_embed = extra_options['r_embed']
        clip = extra_options['clip']
        cnet = extra_options['cnet']

        # generate lr_guide when x_lr is set but lr_guide is None
        x_lr = self.x_lr.to(dtype=x.dtype, device=x.device)
        x_lr = extra_options['model'].embedding(x_lr)
        
        r_embed = extra_options['model'].gen_r_embedding(r).to(dtype=x.dtype)
        for c in extra_options['model'].t_conds:
            t_cond = extra_options.get(c, torch.zeros_like(r))
            r_embed = torch.cat([r_embed, extra_options['model'].gen_r_embedding(t_cond).to(dtype=x.dtype)], dim=1)
        clip = extra_options['model'].gen_c_embeddings(clip_text, clip_text_pooled, clip_img)
        
        level_outputs, lr_enc = extra_options['model']._down_encode(
            x_lr, r_embed, clip, cnet, require_q=True, lr_guide=None)
        _, lr_dec = extra_options['model']._up_decode(
            level_outputs, r_embed, clip, cnet, require_ff=True, agg_f=None)
        
        return ([f.chunk(2)[0].repeat(2, 1, 1, 1) for f in lr_enc], 
                [f.chunk(2)[0].repeat(2, 1, 1, 1) for f in lr_dec])

    def apply_guide(self, x, lr_guide, guide_weight, extra_options):
        r_embed = extra_options['r_embed']
        clip = extra_options['clip']
        cnet = extra_options['cnet']
        pag_patch_flag = extra_options['pag_patch_flag']
        sag_func = extra_options['sag_func']
        guide_mode_weighted = self.guide_type == "weighted"

        r_embed = extra_options['model'].gen_r_embedding(extra_options['r']).to(dtype=x.dtype)
        for c in extra_options['model'].t_conds:
            t_cond = extra_options.get(c, torch.zeros_like(extra_options['r']))
            r_embed = torch.cat([r_embed, extra_options['model'].gen_r_embedding(t_cond).to(dtype=x.dtype)], dim=1)        
        clip = extra_options['model'].gen_c_embeddings(extra_options['clip_text'], extra_options['clip_text_pooled'], extra_options['clip_img'])
        
        level_outputs = extra_options['model']._down_encode(
            x, r_embed, clip, cnet, 
            require_q=False, 
            r_emb_lite=extra_options['model'].gen_r_embedding(extra_options['r']),  
            guide_weight=guide_weight,
            guide_mode_weighted=guide_mode_weighted,
            lr_guide=lr_guide[0] if lr_guide is not None else None,
            pag_patch_flag=pag_patch_flag,
        )

        x = extra_options['model']._up_decode(
            level_outputs, r_embed, clip, cnet, 
            require_ff=False, 
            r_emb_lite=extra_options['model'].gen_r_embedding(extra_options['r']), 
            guide_weight=guide_weight,
            guide_mode_weighted=guide_mode_weighted,
            agg_f=lr_guide[1] if lr_guide is not None else None,
            pag_patch_flag=pag_patch_flag,
            sag_func=sag_func,
        )

        return x
    
    def to(self, device):
        if self.x_lr is not None:
            self.x_lr = self.x_lr.to(device)
        if self.guide_weights is not None:
            self.guide_weights = self.guide_weights.to(device)
        if self.lr_guide is not None:
            self.lr_guide = ([f.to(device) for f in self.lr_guide[0]], [f.to(device) for f in self.lr_guide[1]])
        return self


class StageUP(StageC):
    def __init__(self, c_in=16, c_out=16, c_r=64, patch_size=1, c_cond=2048, c_hidden=[2048, 2048], nhead=[32, 32],
                 blocks=[[8, 24], [24, 8]], block_repeat=[[1, 1], [1, 1]], level_config=['CTA', 'CTA'],
                 c_clip_text=1280, c_clip_text_pooled=1280, c_clip_img=768, c_clip_seq=4, kernel_size=3,
                 dropout=[0.0, 0.0], self_attn=True, t_conds=['sca', 'crp'], switch_level=[False], stable_cascade_stage=None,
                 dtype=None, device=None, operations=None):

        self.c_hidden = c_hidden
        self.blocks = blocks
            
        super().__init__(c_in=c_in, c_out=c_out, c_r=c_r, patch_size=patch_size, c_cond=c_cond, c_hidden=c_hidden, nhead=nhead,
                 blocks=blocks, block_repeat=block_repeat, level_config=level_config,
                 c_clip_text=c_clip_text, c_clip_text_pooled=c_clip_text_pooled, c_clip_img=c_clip_img, c_clip_seq=c_clip_seq, kernel_size=kernel_size,
                 dropout=dropout, self_attn=self_attn, t_conds=t_conds, switch_level=switch_level, stable_cascade_stage=stable_cascade_stage,
                 dtype=dtype, device=device, operations=operations)
        
        self._init_extra_parameter()

    def _init_extra_parameter(self):
        self.agg_net, self.agg_net_up, self.norm_down_blocks, self.norm_up_blocks = (nn.ModuleList() for _ in range(4))

        for _ in range(2):                 
            self.agg_net.append(TransInr(time_dim=self.c_r))    #ind=2048, ch=1024, n_head=32, n_groups=64, f_dim=1024... head_dim = 32 
            self.agg_net_up.append(TransInr(time_dim=self.c_r)) 

        for i in range(len(self.c_hidden)):
            up_blocks = nn.ModuleList()
            for j in range(self.blocks[0][i]):
                if j % 4 == 0:
                    up_blocks.append(ScaleNormalize_res(self.c_hidden[0], self.c_r, conds=[]))
            self.norm_down_blocks.append(up_blocks)

        for i in reversed(range(len(self.c_hidden))):
            up_block = nn.ModuleList()
            for j in range(self.blocks[1][::-1][i]):
                if j % 4 == 0:
                    up_block.append(ScaleNormalize_res(self.c_hidden[0], self.c_r, conds=[]))
            self.norm_up_blocks.append(up_block)



    def _down_encode(self, x, r_embed, clip, cnet=None, require_q=False, lr_guide=None, r_emb_lite=None, guide_weight=1, guide_mode_weighted=False, pag_patch_flag=False):
    
        if require_q:
            qs = []
        
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
        
        for stage_cnt, (down_block, downscaler, repmap) in enumerate(block_group):
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for inner_cnt, block in enumerate(down_block):

                    if isinstance(block, ResBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, ResBlock)):
                        if cnet is not None and len(cnet) != 0:# and lr_guide is None:
                            next_cnet = cnet.pop()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(next_cnet, size=x.shape[-2:], mode="bilinear", align_corners=True).to(dtype=x.dtype)
                        x = block(x)
                        
                    elif isinstance(block, AttnBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, AttnBlock)):
                        x = block(x, clip)
                        if require_q and (inner_cnt == 2):
                            qs.append(x.clone())
                        if lr_guide is not None and (inner_cnt == 2):
                            guide = self.agg_net[stage_cnt](x.shape, x, lr_guide[stage_cnt], r_emb_lite)
                            
                            guide_flat = torch.zeros_like(x)
                            for i2 in range(0, len(guide)): 
                                guide_flat = guide_flat + guide[i2].unsqueeze(0)
                            guide_flat = guide_flat / len(guide)
                            
                            x = x + guide_flat.to(dtype=x.dtype)

                    elif isinstance(block, TimestepBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, TimestepBlock)):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                        
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)  # 0 indicate last output
        if require_q:
            return level_outputs, qs
        return level_outputs



    def _up_decode(self, level_outputs, r_embed, clip, cnet=None, require_ff=False, agg_f=None, r_emb_lite=None, guide_weight=1, guide_mode_weighted=False, pag_patch_flag=False, sag_func=None):

        if require_ff:
            agg_feas = []
            
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):

                    if isinstance(block, ResBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, ResBlock)):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                            x = torch.nn.functional.interpolate(x, skip.shape[-2:],
                                mode="bilinear", align_corners=True).to(dtype=x.dtype)
                            
                        if cnet is not None and len(cnet) != 0 and agg_f is None:
                            next_cnet = cnet.pop()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(next_cnet, size=x.shape[-2:], mode="bilinear",align_corners=True).to(dtype=x.dtype)
                        x = block(x, skip)
                        
                    elif isinstance(block, AttnBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, AttnBlock)):
                        #x = block(x, clip)
                        if i == 0 and j == 0 and k == 2 and sag_func is not None: 
                            orig_x = x

                            extra_options={}
                            extra_options["n_heads"] = 32 #self.n_heads  32 * 64 = 2048... q v k are b,seq_len,2048
                            extra_options["dim_head"] = 64 #self.d_head
                            extra_options["attn_precision"] = None # self.attn_precision
                            extra_options["cond_or_uncond"] = [1,0] # self.attn_precision
                            
                            x = block.norm(x)
                            kv = block.kv_mapper(clip)
                            orig_shape = x.shape
                            x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4

                            kv = torch.cat([x, kv], dim=1) # unrolled attention blocks here
                            q_ = block.attention.attn.to_q(x)
                            k_ = block.attention.attn.to_k(kv)
                            v_ = block.attention.attn.to_v(kv)
                            x = sag_func(q_, k_, v_, extra_options)
                            x = block.attention.attn.out_proj(x)
                            x = orig_x + x.permute(0, 2, 1).view(*orig_shape)
                        elif i == 0 and j ==0 and k == 2 and pag_patch_flag == True and sag_func == None: # and require_ff == False: # and agg_f is None:
                            #x = torch.randn_like(x).to('cuda')

                            #gaussian_blur = torchvision.transforms.GaussianBlur(3, sigma=2.0)
                            #x = gaussian_blur(x)
                            x = block(torch.randn_like(x).to('cuda'), clip) #random query
                            #x = torch.randn_like(x).to('cuda')
                            #x = block(torch.full_like(x, 0.0).to('cuda'), clip)"""
                        else:
                            x = block(x, clip)
                            
                        if require_ff and (k == 2):
                            agg_feas.append(x.clone())
                        if agg_f is not None and (k == 2):
                            guide = self.agg_net_up[i](x.shape, x, agg_f[i], r_emb_lite) 
                            
                            guide_flat = torch.zeros_like(x)
                            for i2 in range(0, len(guide)): 
                                guide_flat = guide_flat + guide[i2].unsqueeze(0)
                            guide_flat = guide_flat / len(guide)

                            if guide_mode_weighted is True:
                                x = (1 - guide_weight) * x + guide_weight * guide_flat.to(dtype=x.dtype)
                            else:
                                x = x + guide_weight * guide_flat.to(dtype=x.dtype)

                    elif isinstance(block, TimestepBlock) or (hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, TimestepBlock)):
                        x = block(x, r_embed)
                        
                    else:
                        x = block(x)
                        
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)

        if require_ff:
            return x, agg_feas

        return x



    def forward(self, x, r, clip_text, clip_text_pooled, clip_img, control=None, **kwargs):
        to = kwargs.get("transformer_options", {}).copy()
        patch = to.get("patches_replace", {}).get("ultracascade", {}).get("main", None)

        r_embed = self.gen_r_embedding(r).to(dtype=x.dtype)
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.zeros_like(r))
            r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond).to(dtype=x.dtype)], dim=1)
        clip = self.gen_c_embeddings(clip_text, clip_text_pooled, clip_img)

        x = self.embedding(x)
        
        if control is not None:
            cnet = control.get("input")
        else:
            cnet = None

        pag_patch_flag = False
        if "patches_replace" in to and "pag_patch" in to["patches_replace"]:
            pag_patch_flag = True

        sag_func = to.get('patches_replace', {}).get('attn1', {}).get(('middle', 0, 0), None)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if patch and patch.x_lr is not None:
                extra_options = {
                    'model': self,
                    'r': r,
                    'r_embed': r_embed,
                    'clip': clip,
                    'clip_text': clip_text,
                    'clip_text_pooled': clip_text_pooled,
                    'clip_img': clip_img,
                    'cnet': cnet,
                    'sigmas': to.get('sigmas', torch.tensor([999999999.9])),
                    'pag_patch_flag': pag_patch_flag,
                    'sag_func': sag_func,
                }
                x = patch(x, r, clip_text, clip_text_pooled, clip_img, extra_options)
            else:
                level_outputs = self._down_encode(x, r_embed, clip, cnet)
                x = self._up_decode(level_outputs, r_embed, clip, cnet)

            if x.dtype == torch.float32:
                x = x.to(torch.bfloat16)
        
        return self.clf(x)