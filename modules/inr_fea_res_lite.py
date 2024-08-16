# Code adapted from https://github.com/catcathh/UltraPixel/

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
import numpy as np
from ..modules.common import Linear, LayerNorm2d, TimestepBlock, TransformerEncoder, ImgrecTokenizer, ScaleNormalize_res

from einops import rearrange

def batched_linear_mm(x, wb):
    # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    return torch.matmul(torch.cat([x, one], dim=-1), wb)

def make_coord_grid(shape, range, device=None):
    # Args: shape: tuple
    #       range: [minv, maxv] or [[minv_1, maxv_1], ..., [minv_d, maxv_d]] for each dim
    # Ret:  grid: shape (*shape, )
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s
        if isinstance(range[0], list) or isinstance(range[0], tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        l = minv + (maxv - minv) * l
        l_lst.append(l)
    grid = torch.meshgrid(*l_lst, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    return grid

def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()

class HypoMlp(nn.Module):
    def __init__(self, depth, in_dim, out_dim, hidden_dim, use_pe, pe_dim, out_bias=0, pe_sigma=1024):
        super().__init__()
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        self.depth = depth
        self.param_shapes = dict()
        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim
        for i in range(depth):  # for each layer the weight
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            self.param_shapes[f"wb{i}"] = (last_dim + 1, cur_dim)
            last_dim = cur_dim
        self.relu = nn.ReLU()
        self.params = None
        self.out_bias = out_bias

    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def forward(self, x):
        B, query_shape = x.shape[0], x.shape[1:-1]
        x = x.view(B, -1, x.shape[-1])
        if self.use_pe:
            x = self.convert_posenc(x)
        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f"wb{i}"])
            if i < self.depth - 1:
                x = self.relu(x)
            else:
                x = x + self.out_bias
        x = x.view(B, *query_shape, -1)
        return x

class TransInr(nn.Module):
    def __init__(self, ind=2048, ch=1024, n_head=32, n_groups=64, f_dim=1024, time_dim=2048, t_conds=[]): # head_dim=64 if f_dim=768 and n_head=12, 32 if 768 and 32
        super().__init__()
        
        self.input_layer = nn.Conv2d(ind, ch, 1)
        self.tokenizer = ImgrecTokenizer(dim=ch, img_channels=ch)

        self.hyponet = HypoMlp(depth=2, in_dim=2, out_dim=ch, hidden_dim=f_dim, use_pe=True, pe_dim=128)
        self.transformer_encoder = TransformerEncoder(dim=f_dim, depth=1, n_head=n_head, head_dim=f_dim // n_head, ff_dim=f_dim)
        self.base_params = nn.ParameterDict()
        
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()
        
        for name, shape in self.hyponet.param_shapes.items():
            self.base_params[name] = nn.Parameter(init_wb(shape))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(f_dim),
                nn.Linear(f_dim, shape[0] - 1),
            )
            
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
            
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, f_dim))
        self.output_layer = nn.Conv2d(ch, ind, 1)
        self.mapp_t = TimestepBlock(ind, time_dim, conds=t_conds)
        self.hr_norm = ScaleNormalize_res(ind, 64, conds=[])
        self.normalize_final = LayerNorm2d(ind, elementwise_affine=False, eps=1e-6)

        self.toout = nn.Sequential(
            Linear(ind * 2, ind // 4), 
            nn.GELU(), 
            Linear(ind // 4, ind)
        )

        mask = torch.zeros((1, 1, 32, 32))
        h, w = 32, 32
        center_h, center_w = h // 2, w // 2
        low_freq_h, low_freq_w = h // 4, w // 4
        mask[
            :,
            :,
            center_h - low_freq_h : center_h + low_freq_h,
            center_w - low_freq_w : center_w + low_freq_w,
        ] = 1
        self.mask = mask


    def forward(self, target_shape, target, dtokens, t_emb):

        original = dtokens

        dtokens = self.input_layer(dtokens)
        dtokens = self.tokenizer(dtokens)

        wtokens = einops.repeat(self.wtokens, "n d -> b n d", b=dtokens.shape[0])

        trans_out = self.transformer_encoder(torch.cat([dtokens, wtokens], dim=1))
        trans_out = trans_out[:, -len(self.wtokens) :, :]

        params = dict()
        for name, shape in self.hyponet.param_shapes.items():
            wb = einops.repeat(self.base_params[name], "n m -> b n m", b=dtokens.shape[0])
            w, b = wb[:, :-1, :], wb[:, -1:, :]

            l, r = self.wtoken_rng[name]
            x = self.wtoken_postfc[name](trans_out[:, l:r, :])
            x = x.transpose(-1, -2)  # (B, shape[0] - 1, g)
            w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)

            wb = torch.cat([w, b], dim=1)
            params[name] = wb
        coord = make_coord_grid(target_shape[2:], (-1, 1), device=dtokens.device)
        coord = einops.repeat(coord, "h w d -> b h w d", b=dtokens.shape[0])
        self.hyponet.set_params(params)
        ori_up = F.interpolate(original.float(), target_shape[2:])
        hr_rec = (self.output_layer(rearrange(self.hyponet(coord), "b h w c -> b c h w")) + ori_up)

        #output = self.toout(torch.cat((hr_rec, target), dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if target.shape[0] == 1:
            target = target.repeat(hr_rec.shape[0],1,1,1)
        #target = target.repeat(2 // hr_rec.shape[0],1,1,1)
        output = self.toout(torch.cat((hr_rec, target), dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2) #changed this line to accomodate multiple clip_img
        
        output = self.mapp_t(output, t_emb)
        output = self.normalize_final(output)
        output = self.hr_norm(output)

        return output

