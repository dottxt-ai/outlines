from typing import TYPE_CHECKING, Callable

from llama_index.callbacks import CallbackManager
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.llms.base import LLM
from outlines.tools.llama_index import set_llama_index_model_function

if TYPE_CHECKING:
    from llama_index.core import BaseQueryEngine


class BaseModel:

    def __init__(self, llama_index_engine: "BaseQueryEngine" = None, *args, **kwargs):
        self.llama_index_engine = llama_index_engine

    def run_with_llama_index(self, prompt: str, func: Callable) -> str:
        """Run through the llama_index engine the outlines function with the user prompt"""
        set_llama_index_model_function(func)
        response = self.llama_index_engine.query(prompt)
        return response
