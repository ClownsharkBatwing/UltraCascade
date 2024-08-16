# Code adapted from https://github.com/Stability-AI/StableCascade, https://github.com/comfyanonymous/ComfyUI/, https://github.com/catcathh/UltraPixel/

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math

class Linear(torch.nn.Linear):
    def reset_parameters(self):
        return None


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TimestepBlock(nn.Module):
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
        return x * (1 + a) + b
    
    
class UpDownBlock2d(nn.Module):
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
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_head, head_dim, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim
        self.to_q = nn.Sequential(nn.SiLU(), Linear(dim, inner_dim))
        self.to_kv = nn.Sequential(nn.SiLU(), Linear(dim, inner_dim * 2))
        self.scale = head_dim**-0.5

    def forward(self, fr, to=None):
        if to is None:
            to = fr
        q = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_head),
            [q, k, v],
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)  # b h n n
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return out


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
                        PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
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
