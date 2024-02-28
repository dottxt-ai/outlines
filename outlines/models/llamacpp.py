from typing import TYPE_CHECKING, List, Optional, Union

import torch

from outlines.integrations.llamacpp import (  # noqa: F401
    CFGLogitsProcessor,
    JSONLogitsProcessor,
    LlamaCppTokenizer,
    LogitsProcessor,
    RegexLogitsProcessor,
)

if TYPE_CHECKING:
    from llama_cpp import Llama


class LlamaCpp:
    """Represents a `llama_cpp` model."""

    def __init__(self, model: "Llama"):
        self.model = model
        self.tokenizer = LlamaCppTokenizer(model=model)


def llamacpp(model_path: str, device: Optional[str] = None, **model_kwargs) -> LlamaCpp:
    from llama_cpp import Llama

    if device == "cuda":
        model_kwargs["n_gpu_layers"].setdefault(-1)

    model = Llama(model_path, **model_kwargs)
    return LlamaCpp(model=model)


class LlamaSequenceGenerator:
    def __init__(self, logits_processor: Optional[LogitsProcessor], model: LlamaCpp):
        self.model = model.model
        self.logits_processor = logits_processor

    def __call__(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        rng: Optional[torch.Generator] = None,
        **model_kwargs,
    ) -> Union[str, List[str]]:
        from llama_cpp import LogitsProcessorList

        if isinstance(prompts, str):
            prompts = [prompts]

        if rng is None:
            rng = torch.Generator(device="cpu")
            rng.seed()

        results = []
        for prompt in prompts:
            processors = []
            if self.logits_processor is not None:
                processors = [self.logits_processor.copy()]

            completion = self.model.create_completion(
                prompt,
                max_tokens=max_tokens,
                stop=stop_at,
                seed=rng.initial_seed(),
                logits_processor=LogitsProcessorList(processors),
                **model_kwargs,
            )
            assert isinstance(completion, dict)
            result = completion["choices"][0]["text"]
            results.append(result)

            self.model.reset()

        formatted = [self.format_sequence(sequence) for sequence in results]

        return formatted if len(formatted) > 1 else formatted[0]

    def format_sequence(self, sequence: str) -> str:
        """Translate the generated sequence to another type.

        This method is for instance overridden when generating JSON to either
        return a dictionnary or a Pydantic model.

        Parameters
        ----------
        sequence
            A generated sequences.

        Returns
        -------
        The formatted sequence.
        """
        return sequence

    def stream(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        rng: Optional[torch.Generator] = None,
    ):
        raise NotImplementedError(
            "Streaming is not implemented for the `llama.cpp` integration."
        )
