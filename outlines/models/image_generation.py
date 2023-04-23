"""Router for image generation models."""
from .hf_diffusers import HuggingFaceDiffuser
from .openai import OpenAIImageGeneration

hf = HuggingFaceDiffuser
openai = OpenAIImageGeneration
