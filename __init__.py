import importlib

modules_to_import = [
    'UltraCascade.nodes',
    'UltraCascade.nodes_sag_rag',
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in modules_to_import:
    module = importlib.import_module(module_name)
    
    if hasattr(module, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    
    if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
