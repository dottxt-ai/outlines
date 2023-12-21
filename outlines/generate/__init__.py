import torch
if torch.backends.mps.is_available():
    from .api_mlx import cfg, choice, format, json, regex, text
else: 
    from .api import cfg, choice, format, json, regex, text



