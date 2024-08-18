from comfy.ldm.cascade.stage_b import StageB
from comfy.ldm.cascade.stage_c import StageC
from .modules.stage_b2 import StageB2
from .modules.stage_up import StageUP
import types

class UltraCascade_StageB_Patcher:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"

    CATEGORY = "advanced/loaders"

    def main(self, model):

        for attr_name in dir(StageB2):
            attr = getattr(StageB2, attr_name)
            if callable(attr) and not attr_name.startswith('__'):
                setattr(model.model.diffusion_model, attr_name, types.MethodType(attr, model.model.diffusion_model))

        model.model.diffusion_model.set_effnet_batch(None)
        model.model.diffusion_model.set_effnet_batch_maps(None)

        return (model,)
    
class UltraCascade_StageB_Unpatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"

    CATEGORY = "advanced/loaders"

    def main(self, model):

        for attr_name in dir(StageB2):
            if hasattr(StageB, attr_name):
                attr = getattr(StageB, attr_name)
                if callable(attr) and not attr_name.startswith('__'):
                    setattr(model.model.diffusion_model, attr_name, types.MethodType(attr, model.model.diffusion_model))
            else:
                delattr(model.model.diffusion_model, attr_name)

        return (model,)

class UltraCascade_StageC_Patcher:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"

    CATEGORY = "advanced/loaders"

    def main(self, model):

        for attr_name in dir(StageUP):
            attr = getattr(StageUP, attr_name)
            if callable(attr) and not attr_name.startswith('__'):
                setattr(model.model.diffusion_model, attr_name, types.MethodType(attr, model.model.diffusion_model))
        
        model.model.diffusion_model.x_lr=None
        model.model.diffusion_model.lr_guide=None 
        model.model.diffusion_model.require_f=False 
        model.model.diffusion_model.require_t=False 
        model.model.diffusion_model.guide_weight=1.0
        model.model.diffusion_model.guide_weights=None
        model.model.diffusion_model.guide_weights_tmp=None
        model.model.diffusion_model.guide_mode_weighted=False
        
        model.model.diffusion_model._init_extra_parameter()
        
        return (model,)
    
class UltraCascade_StageC_Unpatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"

    CATEGORY = "advanced/loaders"

    def main(self, model):

        for attr_name in dir(StageUP):
            if hasattr(StageC, attr_name):
                attr = getattr(StageC, attr_name)
                if callable(attr) and not attr_name.startswith('__'):
                    setattr(model.model.diffusion_model, attr_name, types.MethodType(attr, model.model.diffusion_model))
            else:
                delattr(model.model.diffusion_model, attr_name)

        return (model,)


NODE_CLASS_MAPPINGS = {
    "UltraCascade_StageB_Patcher": UltraCascade_StageB_Patcher,
    "UltraCascade_StageB_Unpatcher": UltraCascade_StageB_Unpatcher,
    "UltraCascade_StageC_Patcher": UltraCascade_StageC_Patcher,
    "UltraCascade_StageC_Unpatcher": UltraCascade_StageC_Unpatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraCascade_StageB_Patcher": "UltraCascade Stage B Patcher",
    "UltraCascade_StageB_Unpatcher": "UltraCascade Stage B Unpatcher",
    "UltraCascade_StageC_Patcher": "UltraCascade Stage C Patcher",
    "UltraCascade_StageC_Unpatcher": "UltraCascade Stage C Unpatcher",
}

