from .modules.stage_b2 import StageB2
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

NODE_CLASS_MAPPINGS = {
    "UltraCascade_StageB_Patcher": UltraCascade_StageB_Patcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraCascade_StageB_Patcher": "UltraCascade Stage B Patcher",
}
