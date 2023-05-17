"""Router for text completion models."""
from .hf_transformers import HuggingFaceCompletion
from .openai import OpenAICompletion

hf = HuggingFaceCompletion
openai = OpenAICompletion
