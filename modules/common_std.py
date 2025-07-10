# Code adapted from https://github.com/Stability-AI/StableCascade, https://github.com/comfyanonymous/ComfyUI/, https://github.com/catcathh/UltraPixel/

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math

from comfy.ldm.modules.attention import optimized_attention, attention_pytorch
import comfy.ops

class Linear(torch.nn.Linear):
    def reset_parameters(self):
        return None


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)








class ReOptimizedAttention_fp32(nn.Module):
    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = nhead

        self.to_q = operations.Linear(c, c, bias=True, dtype=dtype, device=device)
        self.to_k = operations.Linear(c, c, bias=True, dtype=dtype, device=device)
        self.to_v = operations.Linear(c, c, bias=True, dtype=dtype, device=device)

        self.out_proj = operations.Linear(c, c, bias=True, dtype=dtype, device=device)

    def forward(self, q, k, v, style_block=None, attn_mask=None): # k,v always identical
        dtype_init = q.dtype
        
        q = F.linear(q.float(), self.to_q.weight.data.float(), self.to_q.bias.data.float())
        k = F.linear(k.float(), self.to_k.weight.data.float(), self.to_k.bias.data.float())
        v = F.linear(v.float(), self.to_v.weight.data.float(), self.to_v.bias.data.float())
        
        q = style_block(q, "q_proj")
        k = style_block(k, "k_proj")
        v = style_block(v, "v_proj")

        if attn_mask is not None:
            out = attention_pytorch(q, k, v, self.heads, mask=attn_mask)
        else:
            out = optimized_attention(q, k, v, self.heads)        
        #out = optimized_attention(q.float(), k.float(), v.float(), self.heads).to(q)
        
        out = style_block(out, "attn")

        out = self.out_proj(out)
        #out = F.linear(out.float(), self.out_proj.weight.data.float(), self.out_proj.bias.data.float())
        
        out = style_block(out, "out")
        return out.to(dtype_init)

class ReAttention2D_fp32(nn.Module):
    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.attn = ReOptimizedAttention(c, nhead, dtype=dtype, device=device, operations=operations)
        # self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True, dtype=dtype, device=device)

    def forward(self, x, kv, self_attn=False, style_block=None, attn_mask=None):
        dtype_init = x.dtype
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        
        x  = x.float()
        kv = kv.float()
        
        x = style_block(x, "attn_norm")
        if self_attn:
            kv = torch.cat([x, kv], dim=1)
        else:
            pass
        # x = self.attn(x, kv, kv, need_weights=False)[0]
        x = self.attn(x, kv, kv, style_block=style_block, attn_mask=attn_mask)
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x.to(dtype_init)

class ReAttnBlock_fp32(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d_op(operations)(c, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.attention = ReAttention2D(c, nhead, dropout, dtype=dtype, device=device, operations=operations)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            operations.Linear(c_cond, c, dtype=dtype, device=device)
        )

    def forward(self, x, kv, style_block=None, attn_mask=None):
        dtype_init = x.dtype
        x  = x.float()
        kv = kv.float()
        
        kv = self.kv_mapper[0](kv)
        kv = F.linear(kv, self.kv_mapper[1].weight.data.float(), self.kv_mapper[1].bias.data.float())
        
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn, style_block=style_block, attn_mask=attn_mask)
        
        return x.to(dtype_init)





class ReTimestepBlock_fp32(nn.Module):
    def __init__(self, c, c_timestep, conds=['sca'], dtype=None, device=None, operations=None):
        super().__init__()
        self.mapper = operations.Linear(c_timestep, c * 2, dtype=dtype, device=device)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", operations.Linear(c_timestep, c * 2, dtype=dtype, device=device))

    def forward(self, x, t):
        dtype_init = x.dtype
        
        x = x.float()
        t = t.float()
        
        t = t.chunk(len(self.conds) + 1, dim=1)
        
        t_out = F.linear(t[0], self.mapper.weight.data.float(), self.mapper.bias.data.float())
        
        a, b = t_out[:, :, None, None].chunk(2, dim=1)
        
        #a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        
        
        for i, c in enumerate(self.conds):
            t_out = F.linear(t[i+1], getattr(self, f"mapper_{c}").weight.data, getattr(self, f"mapper_{c}").bias.data)
            ac, bc = t_out[:, :, None, None].chunk(2, dim=1)
            #ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(2, dim=1)
            a, b = a + ac, b + bc
            
        output = x * (1 + a) + b
        return output #.to(dtype_init)


class ReUpDownBlock2d_fp32(nn.Module):
    def __init__(self, c_in, c_out, mode, enabled=True, dtype=None, device=None, operations=None):
        super().__init__()
        assert mode in ['up', 'down']
        interpolation = nn.Upsample(scale_factor=2 if mode == 'up' else 0.5, mode='bilinear',
                                    align_corners=True) if enabled else nn.Identity()
        mapping = operations.Conv2d(c_in, c_out, kernel_size=1, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([interpolation, mapping] if mode == 'up' else [mapping, interpolation])

    def forward(self, x):
        dtype_init = x.dtype
        
        x = x.float()
        
        #x = F.conv2d(x, self.blocks[1].weight.data.float(), block[1].bias.data.float())

        for block in self.blocks:
            if type(block) == nn.Identity:
                continue
            x = F.conv2d(x, block.weight.data.float(), block.bias.data.float())
            #x = block(x)
        
        return x.to(dtype_init)








class ReOptimizedAttention(nn.Module):
    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = nhead

        self.to_q = operations.Linear(c, c, bias=True, dtype=dtype, device=device)
        self.to_k = operations.Linear(c, c, bias=True, dtype=dtype, device=device)
        self.to_v = operations.Linear(c, c, bias=True, dtype=dtype, device=device)

        self.out_proj = operations.Linear(c, c, bias=True, dtype=dtype, device=device)

    def forward(self, q, k, v, style_block=None, attn_mask=None): # k,v always identical
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        dtype_init = q.dtype
        
        #q = F.linear(q.float(), self.to_q.weight.data.float(), self.to_q.bias.data.float())
        #k = F.linear(k.float(), self.to_k.weight.data.float(), self.to_k.bias.data.float())
        #v = F.linear(v.float(), self.to_v.weight.data.float(), self.to_v.bias.data.float())
        
        q = style_block(q, "q_proj")
        k = style_block(k, "k_proj")
        v = style_block(v, "v_proj")

        if attn_mask is not None:
            out = attention_pytorch(q, k, v, self.heads, mask=attn_mask)
        else:
            out = optimized_attention(q, k, v, self.heads)
        #out = optimized_attention(q.float(), k.float(), v.float(), self.heads).to(q)
        
        out = style_block(out, "attn")

        out = self.out_proj(out)
        #out = F.linear(out.float(), self.out_proj.weight.data.float(), self.out_proj.bias.data.float())
        
        out = style_block(out, "out")
        return out #.to(dtype_init)

class ReAttention2D(nn.Module):
    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.attn = ReOptimizedAttention(c, nhead, dtype=dtype, device=device, operations=operations)
        # self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True, dtype=dtype, device=device)

    def forward(self, x, kv, self_attn=False, style_block=None, attn_mask=None):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        x = style_block(x, "attn_norm")
        if self_attn:
            kv = torch.cat([x, kv], dim=1)
        else:
            pass
        # x = self.attn(x, kv, kv, need_weights=False)[0]
        x = self.attn(x, kv, kv, style_block=style_block, attn_mask=attn_mask)
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x

class ReAttnBlock(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d_op(operations)(c, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.attention = ReAttention2D(c, nhead, dropout, dtype=dtype, device=device, operations=operations)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            operations.Linear(c_cond, c, dtype=dtype, device=device)
        )

    def forward(self, x, kv, style_block=None, attn_mask=None):
        kv = self.kv_mapper(kv)
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn, style_block=style_block, attn_mask=attn_mask)
        return x


def LayerNorm2d_op(operations):
    class LayerNorm2d(operations.LayerNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, x):
            return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return LayerNorm2d


class GlobalResponseNorm(nn.Module):
    "from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105"
    def __init__(self, dim, dtype=None, device=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(1, 1, 1, dim, dtype=dtype, device=device))
        self.beta = nn.Parameter(torch.empty(1, 1, 1, dim, dtype=dtype, device=device))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return comfy.ops.cast_to_input(self.gamma, x) * (x * Nx) + comfy.ops.cast_to_input(self.beta, x) + x


class ResBlock(nn.Module):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0, dtype=None, device=None, operations=None):  # , num_heads=4, expansion=2):
        super().__init__()
        self.depthwise = operations.Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c, dtype=dtype, device=device)
        #         self.depthwise = SAMBlock(c, num_heads, expansion)
        self.norm = LayerNorm2d_op(operations)(c, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.channelwise = nn.Sequential(
            operations.Linear(c + c_skip, c * 4, dtype=dtype, device=device),
            nn.GELU(),
            GlobalResponseNorm(c * 4, dtype=dtype, device=device),
            nn.Dropout(dropout),
            operations.Linear(c * 4, c, dtype=dtype, device=device)
        )

    def forward(self, x, x_skip=None):
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res



class FeedForwardBlock(nn.Module):
    def __init__(self, c, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = LayerNorm2d_op(operations)(c, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.channelwise = nn.Sequential(
            operations.Linear(c, c * 4, dtype=dtype, device=device),
            nn.GELU(),
            GlobalResponseNorm(c * 4, dtype=dtype, device=device),
            nn.Dropout(dropout),
            operations.Linear(c * 4, c, dtype=dtype, device=device)
        )

    def forward(self, x):
        x = x + self.channelwise(self.norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class TimestepBlock(nn.Module):
    def __init__(self, c, c_timestep, conds=['sca'], dtype=None, device=None, operations=None):
        super().__init__()
        self.mapper = operations.Linear(c_timestep, c * 2, dtype=dtype, device=device)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", operations.Linear(c_timestep, c * 2, dtype=dtype, device=device))

    def forward(self, x, t):
        t = t.chunk(len(self.conds) + 1, dim=1)
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        for i, c in enumerate(self.conds):
            ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(2, dim=1)
            a, b = a + ac, b + bc
        return x * (1 + a) + b


class UpDownBlock2d(nn.Module):
    def __init__(self, c_in, c_out, mode, enabled=True, dtype=None, device=None, operations=None):
        super().__init__()
        assert mode in ['up', 'down']
        interpolation = nn.Upsample(scale_factor=2 if mode == 'up' else 0.5, mode='bilinear',
                                    align_corners=True) if enabled else nn.Identity()
        mapping = operations.Conv2d(c_in, c_out, kernel_size=1, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([interpolation, mapping] if mode == 'up' else [mapping, interpolation])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


























"""class TimestepBlock(nn.Module):
    def __init__(self, c, c_timestep, conds=['sca']):
        super().__init__()
        self.mapper = Linear(c_timestep, c * 2)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", Linear(c_timestep, c * 2))

    def forward(self, x, t):
        t = t.chunk(len(self.conds) + 1, dim=1)
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        for i, c in enumerate(self.conds):
            ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(2, dim=1)
            a, b = a + ac, b + bc
        return x * (1 + a) + b"""
    
    
"""class UpDownBlock2d(nn.Module):
    def __init__(self, c_in, c_out, mode, enabled=True):
        super().__init__()
        assert mode in ['up', 'down']
        interpolation = nn.Upsample(scale_factor=2 if mode == 'up' else 0.5, mode='bilinear',
                                    align_corners=True) if enabled else nn.Identity()
        mapping = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.blocks = nn.ModuleList([interpolation, mapping] if mode == 'up' else [mapping, interpolation])

    def forward(self, x):
        for block in self.blocks:
            x = block(x.float())
        return x"""


class Attention_original(nn.Module):
    def __init__(self, dim, n_head, head_dim, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim
        self.to_q  = nn.Sequential(nn.SiLU(), Linear(dim, inner_dim))
        self.to_kv = nn.Sequential(nn.SiLU(), Linear(dim, inner_dim * 2))
        self.scale = head_dim**-0.5

    def forward(self, fr, to=None):
        if to is None:
            to = fr
        q       = self.to_q(fr)
        k, v    = self.to_kv(to).chunk(2, dim=-1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_head),
            [q, k, v],
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)  # b h n n
        out  = torch.matmul(attn, v)
        out  = einops.rearrange(out, "b h n d -> b n (h d)")
        return out
    
class Attention(nn.Module):
    def __init__(self, dim, n_head, head_dim, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        inner_dim   = n_head * head_dim
        self.to_q   = nn.Sequential(nn.SiLU(), nn.Linear(dim, inner_dim))
        self.to_kv  = nn.Sequential(nn.SiLU(), nn.Linear(dim, inner_dim * 2))

    def forward(self, fr, to=None):
        if to is None:
            to = fr
        q    = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1)
        x    = optimized_attention(q, k, v, heads=self.n_head)
        return x
        """
        # -> [batch, heads, seq_len, head_dim]
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_head), [q, k, v])
        
        # note: scaled dot product attention: applies scaling factor 1/sqrt(d) internally
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        # -> [batch, seq_len, inner_dim]
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return out
        """

class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0.0):
        super().__init__()
        
        self.net = nn.Sequential(
            Linear(dim, ff_dim),
            nn.GELU(),                #standard cascade version has a GlobalResponseNorm layer after this
            nn.Dropout(dropout),
            Linear(ff_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0.0):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention  (dim, n_head, head_dim, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, ff_dim,           dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for norm_attn, norm_ff in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ff(x)
        return x


class ImgrecTokenizer(nn.Module):
    def __init__(
        self, input_size=32 * 32, patch_size=1, dim=768, padding=0, img_channels=16
    ):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * img_channels, dim)

        self.posemb = nn.Parameter(torch.randn(input_size, dim)) ####HACK ALERT! PATCH OF DEATH
 
    def forward(self, x):
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding)  # (B, C * p * p, L)
        x = x.permute(0, 2, 1).contiguous()

        num_repeats = (x.shape[1] + self.posemb.shape[0] - 1) // self.posemb.shape[0]
        tiled_posemb = self.posemb.repeat(num_repeats, 1)[:x.shape[1]]

        x = self.prefc(x) + tiled_posemb.unsqueeze(0)
        return x


class ScaleNormalize_res(nn.Module):
    def __init__(self, c, scale_c, conds=["sca"]):
        super().__init__()
        self.c_r = scale_c
        self.mapping = TimestepBlock(c, scale_c, conds=conds)
        self.t_conds = conds
        self.alpha = nn.Conv2d(c, c, kernel_size=1)
        self.gamma = nn.Conv2d(c, c, kernel_size=1)
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)

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

    def forward(self, x, std_size=24 * 24):
        scale_val = math.sqrt(math.log(x.shape[-2] * x.shape[-1], std_size))
        scale_val = torch.ones(x.shape[0]).to(x.device) * scale_val
        scale_val_f = self.gen_r_embedding(scale_val)
        for c in self.t_conds:
            t_cond = torch.zeros_like(scale_val)
            scale_val_f = torch.cat([scale_val_f, self.gen_r_embedding(t_cond)], dim=1)

        f = self.mapping(x, scale_val_f)

        return f + x
