import torch
from torch import einsum
import torch.nn.functional as F
import math

from einops import rearrange, repeat
from comfy.ldm.modules.attention import optimized_attention
import comfy.samplers
import comfy.model_patcher


# Self-Attention Guidance code was adapted from the following implementation: https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_sag.py

# from comfy/ldm/modules/attention.py
# but modified to return attention scores as well as output

def attention_basic_with_sim(q, k, v, heads, mask=None, attn_precision=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head ** -0.5

    h = heads
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    # force cast to fp32 to avoid overflowing
    if attn_precision == torch.float32:
        sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return (out, sim)

def rescale_to_0_to_1(x):
    return (x - x.min()) / (x.max() - x.min())

def create_blur_map(x0, attn, sigma=3.0, threshold=1.0, blur_kernel_size=9, mask_mode='hard', clamp=0.0):
    # reshape and GAP the attention map
    _, hw1, hw2 = attn.shape
    b, _, lh, lw = x0.shape

    attn = attn.reshape(b, -1, hw1, hw2)
    mask = attn.mean(1, keepdim=False).sum(1, keepdim=False)     # Global Average Pool
    
    if clamp > 0.0:
        mask = torch.clamp(mask, 0.0, clamp) # use gradient attention mask instead of binary mask
    elif clamp == -1.0:
        mask = rescale_to_0_to_1(mask)
    
    if mask_mode == 'hard_mean':
        mask = mask > mask.mean() + threshold * (mask.max() - mask.mean())
    if mask_mode == 'hard_median':
        mask = mask > mask.median() + threshold * (mask.max() - mask.mean())
    if mask_mode == 'hard_threshold':
        mask = mask > threshold
    if mask_mode == 'soft_threshold':
        mask[mask < threshold] = 0.0
    if mask_mode == 'soft_power':
        mask = mask ** threshold
    if mask_mode == 'raw':
        mask = mask
        
    ratio = 2**(math.ceil(math.sqrt(lh * lw / hw1)) - 1).bit_length()
    mid_shape = [math.ceil(lh / ratio), math.ceil(lw / ratio)]

    mask = mask[:,:hw1] # slice off clip attention that was krazy glued with torch.concat()
    mask = (
        mask.reshape(b, *mid_shape)
        .unsqueeze(1)
        .type(attn.dtype)
    )
    mask = F.interpolate(mask, (lh, lw))    # Upsample

    blurred = gaussian_blur_2d(x0, kernel_size=blur_kernel_size, sigma=sigma)
    blurred = blurred * mask + x0 * (1 - mask)
    return blurred


def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])
    return img


class UltraCascade_SelfAttentionGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "scale": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 5.0, "step": 0.1}),
                             "blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "blur_size": ("INT", {"default": 9, "min": 0, "max": 1000.0, "step": 1}),
                             "mode": (['raw', 'hard_threshold', 'hard_mean', 'hard_median', 'soft_threshold', 'soft_power'], ),
                             "threshold": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                             "clamp": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 1000000000.0, "step": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "UltraCascade"

    def patch(self, model, scale, blur_sigma, blur_size, mode, threshold, clamp):
        m = model.clone()

        blur_kernel_size = blur_size
        mask_mode = mode
        clamp = clamp
        sag_threshold = threshold
        attn_scores = None

        # TODO: make this work properly with chunked batches
        #       currently, we can only save the attn from one UNet call

        def attn_and_record(q, k, v, extra_options):
            nonlocal attn_scores
            # if uncond, save the attention scores
            heads = extra_options["n_heads"]        #ATTN1_REPLACE_PATCH
            cond_or_uncond = extra_options["cond_or_uncond"]
            #b = 1 #q.shape[0] // len(cond_or_uncond)
            b = q.shape[0]
            if 1 in cond_or_uncond:
                uncond_index = cond_or_uncond.index(1)
                # do the entire attention operation, but save the attention scores to attn_scores
                (out, sim) = attention_basic_with_sim(q, k, v, heads=heads, attn_precision=extra_options["attn_precision"])
                # when using a higher batch size, I BELIEVE the result batch dimension is [uc1, ... ucn, c1, ... cn]
                n_slices = heads * b
                attn_scores = sim[n_slices * uncond_index:n_slices * (uncond_index+1)]
                return out
            else:
                return optimized_attention(q, k, v, heads=heads, attn_precision=extra_options["attn_precision"])
        
        def post_cfg_function(args):
            nonlocal blur_kernel_size
            nonlocal mask_mode
            nonlocal attn_scores
            nonlocal clamp
            nonlocal sag_threshold

            uncond_attn = attn_scores

            sag_scale = scale
            sag_sigma = blur_sigma
            sag_threshold = 1.0
            model         = args["model"]
            uncond_pred   = args["uncond_denoised"]
            uncond        = args["uncond"]
            cfg_result    = args["denoised"]
            sigma         = args["sigma"]
            model_options = args["model_options"]
            x             = args["input"]
            if min(cfg_result.shape[2:]) <= 4: #skip when too small to add padding
                return cfg_result
            
            if sag_scale == 0.0:
                return cfg_result

            # create the adversarially blurred image
            degraded = create_blur_map(uncond_pred, uncond_attn, sag_sigma, sag_threshold, blur_kernel_size, mask_mode, clamp) #with bs=6, uncond_pred 6,4,256,256 and uncond_attn (nhead*6 = 144...) 144,256,260
            degraded_noised = degraded + x - uncond_pred
            # call into the UNet
            (sag,) = comfy.samplers.calc_cond_batch(model, [uncond], degraded_noised, sigma, model_options)
            return cfg_result + (degraded - sag) * sag_scale

        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)
        m.set_model_attn1_replace(attn_and_record, "middle", 0, 0)
        return (m, )


class UltraCascade_RandomAttentionGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"

    CATEGORY = "UltraCascade"

    def main(self, model, scale):
        unet_block = "middle"
        unet_block_id = 0
        m = model.clone()

        def random_attention(q, k, v, extra_options, mask=None):
            return None

        def post_cfg_function(args):
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            if scale == 0:
                return cfg_result

            model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, random_attention, "attn1", unet_block, unet_block_id)
            (rag,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

            return cfg_result + (cond_pred - rag) * scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)


class UltraCascade_PerturbedAttentionGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"

    CATEGORY = "UltraCascade"

    def main(self, model, scale):
        unet_block = "middle"
        unet_block_id = 0
        m = model.clone()

        def perturbed_attention(q, k, v, extra_options, mask=None):
            return None

        def post_cfg_function(args):
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            if scale == 0:
                return cfg_result

            model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, perturbed_attention, "attn1_pag", unet_block, unet_block_id)
            (pag,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

            return cfg_result + (cond_pred - pag) * scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)



class UltraCascade_AttentionGuidance_Block:
    def forward(self, blah):
        return None

class UltraCascade_AttentionGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["PAG", "RAG"], ),
                "x_q": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "x_k": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "x_v": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                
                "x_q_eta": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "x_k_eta": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "x_v_eta": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                
                "scale": ("FLOAT", {"default": 3.0, "min": -100.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"

    CATEGORY = "UltraCascade"

    def main(self, model, scale):
        unet_block = "middle"
        unet_block_id = 0
        m = model.clone()

        def random_attention(q, k, v, extra_options, mask=None):
            return None

        def post_cfg_function(args):
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            if scale == 0:
                return cfg_result

            attn_g_block = UltraCascade_AttentionGuidance_Block()

            model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, attn_g_block, "attn1", unet_block, unet_block_id)
            (rag,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

            return cfg_result + (cond_pred - rag) * scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)




NODE_CLASS_MAPPINGS = {
    "UltraCascade_SelfAttentionGuidance": UltraCascade_SelfAttentionGuidance,
    "UltraCascade_RandomAttentionGuidance": UltraCascade_RandomAttentionGuidance,
    "UltraCascade_PerturbedAttentionGuidance": UltraCascade_PerturbedAttentionGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraCascade_SelfAttentionGuidance": "UltraCascade Self-Attention Guidance",
    "UltraCascade_RandomAttentionGuidance": "UltraCascade Random Attention Guidance",
    "UltraCascade_PerturbedAttentionGuidance": "UltraCascade Perturbed Attention Guidance",
}


