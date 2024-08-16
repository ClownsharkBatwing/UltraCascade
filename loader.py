import os

import torch
from torch import nn
import safetensors.torch

import folder_paths
import comfy.latent_formats
import comfy.model_management
import comfy.model_detection
import comfy.supported_models
import comfy.supported_models_base
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import comfy.conds
import comfy.ldm.cascade.common
 
from .modules.stage_up import StageUP


class Null_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


def load_UltraCascade(stage_c_path, stage_up_path):

    sd = comfy.utils.load_torch_file(stage_c_path)
    model = load_unet_state_dict_ultracascade(sd)
    if model is None:
        print("ERROR UNSUPPORTED UNET {}".format(stage_c_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(stage_c_path))
    
    model.model.diffusion_model._init_extra_parameter()
    
    sdd = safetensors.torch.load_file(stage_up_path)         #safely load patches for stage C
    collect_sd = {k: v for k, v in sdd.items()}
    collect_sd = {k[7:] if k.startswith('module.') else k: v for k, v in collect_sd.items()}
    
    train_norm = nn.ModuleList()
    cnt_norm = 0
    for mm in model.model.diffusion_model.modules():
        if isinstance(mm, comfy.ldm.cascade.common.GlobalResponseNorm):
            train_norm.append(Null_Model())
            cnt_norm += 1

    train_norm.append(model.model.diffusion_model.agg_net)
    train_norm.append(model.model.diffusion_model.agg_net_up) 
    train_norm.load_state_dict(collect_sd)
    
    model.model.diffusion_model = model.model.diffusion_model.to(torch.bfloat16)
    
    return model


def load_unet_state_dict_ultracascade(sd): #load unet in diffusers or regular format
    #Allow loading unets from checkpoint files
    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = comfy.utils.calculate_parameters(sd)
    unet_dtype = comfy.model_management.unet_dtype(model_params=parameters)
    load_device = comfy.model_management.get_torch_device()
    model_config = model_config_from_unet_ultracascade(sd, "")

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None: #diffusers mmdit
            model_config = model_config_from_unet_ultracascade(new_sd, "")
            if model_config is None:
                return None
        else: #diffusers unet
            model_config = comfy.model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    print("{} {}".format(diffusers_keys[k], k))

    offload_device = comfy.model_management.unet_offload_device()
    unet_dtype = comfy.model_management.unet_dtype(model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys in unet: {}".format(left_over))
    return comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)


def model_config_from_unet_ultracascade(state_dict, unet_key_prefix, use_base_if_no_match=False):
    unet_config = detect_unet_config_ultracascade(state_dict, unet_key_prefix)
    if unet_config is None:
        return None
    model_config = model_config_from_unet_config_ultracascade(unet_config, state_dict)
    if model_config is None and use_base_if_no_match:
        return comfy.supported_models_base.BASE(unet_config)
    else:
        return model_config


def detect_unet_config_ultracascade(state_dict, key_prefix):
    state_dict_keys = list(state_dict.keys())
    if '{}clf.1.weight'.format(key_prefix) in state_dict_keys: #stable cascade
        unet_config = {}
        text_mapper_name = '{}clip_txt_mapper.weight'.format(key_prefix)
        if text_mapper_name in state_dict_keys:
            unet_config['stable_cascade_stage'] = 'up'
            w = state_dict[text_mapper_name]
            if w.shape[0] == 1536: #stage c lite
                unet_config['c_cond'] = 1536
                unet_config['c_hidden'] = [1536, 1536]
                unet_config['nhead'] = [24, 24]
                unet_config['blocks'] = [[4, 12], [12, 4]]
            elif w.shape[0] == 2048: #stage c full
                unet_config['c_cond'] = 2048
    return unet_config


def model_config_from_unet_config_ultracascade(unet_config, state_dict=None):
    for model_config in models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)

    print("no match {}".format(unet_config))
    return None


class Stable_Cascade_UP(comfy.supported_models.Stable_Cascade_C):
    unet_config = {"stable_cascade_stage": 'up',}

    def get_model(self, state_dict, prefix="", device=None):
        out = StableCascade_UP(self, device=device)
        return out

        
class StableCascade_UP(comfy.model_base.StableCascade_C):
    def __init__(self, model_config, model_type=comfy.model_base.ModelType.STABLE_CASCADE, device=None):
        comfy.model_base.BaseModel.__init__(self, model_config, model_type=model_type, device=device, unet_model=StageUP)
        self.diffusion_model.eval().requires_grad_(False)


models = [Stable_Cascade_UP]

