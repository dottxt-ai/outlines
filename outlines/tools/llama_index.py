from typing import Any

from llama_index.callbacks import CallbackManager
from llama_index.llms.base import CompletionResponse
from llama_index.llms.base import LLM
from llama_index.llms.base import LLMMetadata


llama_index_model_function = None

MODELS_DEFAULT_PARAMS = {
    "default": {
        "context_window": 1024,
        "num_output": 64
    },
    "gpt2": {
        "context_window": 1024,
        "num_output": 64
    }
}


def set_llama_index_model_function(func):
    global llama_index_model_function
    llama_index_model_function = func


def get_llama_index_model_function():
    return llama_index_model_function


class LlamaIndexOutlinesLLM():

    def __init__(self, model_name: str = "default", context_window: int = None, num_output: int = None):
        self.callback_manager = CallbackManager()
        try:
            metadata = MODELS_DEFAULT_PARAMS[model_name]
        except KeyError:
            raise Exception("Invalid model_name")
        metadata.update({"context_window": context_window} if context_window else {})
        metadata.update({"num_output": num_output} if num_output else {})
        self._metadata = LLMMetadata(**metadata)

    @property
    def metadata(self):
        """Values used by llama_index to compute the size of the context text chunks to use in the queries"""
        return self._metadata

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Function called by llama_index to run the query through the LLM, call the outlines function in our case"""
        func = get_llama_index_model_function()
        if func:
            response = func(prompt)
            return CompletionResponse(text=response)
        else:
            raise Exception("The outlines function has not been set")

    ### present because they are abstract_methods of the parent class

    def chat(self, messages, **kwargs: Any):
        """Chat endpoint for LLM."""
        pass
    
    def stream_chat(
        self, messages, **kwargs: Any
    ):
        """Streaming chat endpoint for LLM."""
        pass
    
    def stream_complete(self, prompt: str, **kwargs: Any):
        """Streaming completion endpoint for LLM."""
        pass
    
    async def achat(
        self, messages, **kwargs: Any
    ):
        """Async chat endpoint for LLM."""
        pass
    
    async def acomplete(self, prompt: str, **kwargs: Any):
        """Async completion endpoint for LLM."""
        pass

    async def astream_chat(
        self, messages, **kwargs: Any
    ):
        """Async streaming chat endpoint for LLM."""
        pass
    
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) :
        """Async streaming completion endpoint for LLM."""
        pass
