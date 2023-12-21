import torch
if torch.backends.mps.is_available():
    from .api_mlx import choice, continuation, format, json, regex
else: 
    from .api import choice, continuation, format, json, regex
