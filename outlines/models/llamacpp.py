import math
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray

from outlines.fsm.fsm import CFGFSM, FSM, FSMState, RegexFSM

if TYPE_CHECKING:
    from llama_cpp import Llama


class LlamaSequenceGenerator:
    def __init__(
        self, logits_processor: Optional["LogitsProcessor"], model: "LlamaCpp"
    ):
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

            result = self.model.create_completion(
                prompt,
                max_tokens=max_tokens,
                stop=stop_at,
                seed=rng.initial_seed(),
                logits_processor=LogitsProcessorList(processors),
                **model_kwargs,
            )["choices"][0]["text"]
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


class LlamaCpp:
    """Represents a `llama_cpp` model."""

    def __init__(self, model: "Llama", **kwargs):
        self.model = model
        self.tokenizer = LlamaCppTokenizer(model)


class LlamaCppTokenizer:
    def __init__(self, model, **kwargs):
        self.eos_token_id = model.token_eos()
        self.pad_token_id = self.eos_token_id
        self.special_tokens = {}

        self.vocabulary = {}
        for t in range(model.n_vocab()):
            token_piece = model.tokenizer().decode([t])
            self.vocabulary[token_piece] = t

    def convert_token_to_string(self, token: str) -> str:
        return token


def llamacpp(
    model_path: str,
    device: Optional[str] = None,
    **model_kwargs,
):
    from llama_cpp import Llama

    if device == "cuda":
        model_kwargs["n_gpu_layers"].setdefault(-1)

    model = Llama(model_path, **model_kwargs)

    return LlamaCpp(model)


class LogitsProcessor:
    def __init__(self, tokenizer: LlamaCppTokenizer, fsm: FSM):
        """A FSM-based logits processor.

        Parameters
        ----------
        tokenizer
            An instance of `Tokenizer`
        fsm
            An instance of `FSM`

        """
        self.tokenizer = tokenizer
        self.fsm_state = FSMState(0)
        self.fsm: FSM = fsm
        self.is_first_token = True

    def __call__(
        self, input_ids: NDArray[np.int64], scores: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Use the FSM to bias the logits before sampling the next token."""

        if self.is_first_token:
            self.is_first_token = False
        else:
            last_token = input_ids[-1]
            self.fsm_state = self.fsm.next_state(self.fsm_state, last_token)

        allowed_tokens = self.fsm.allowed_token_ids(self.fsm_state)

        mask = torch.full((scores.shape[-1],), -math.inf, device="cpu").numpy()
        mask[allowed_tokens] = 0
        biased_scores = scores + mask

        return biased_scores

    def copy(self):
        return LogitsProcessor(self.tokenizer, self.fsm.copy())


class RegexLogitsProcessor(LogitsProcessor):
    def __init__(self, regex_string: str, tokenizer: LlamaCppTokenizer):
        """Compile the FSM that drives the regex-guided generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        tokenizer
            An instance of `Tokenizer`

        """
        fsm = RegexFSM(regex_string, tokenizer)
        super().__init__(tokenizer, fsm)


class CFGLogitsProcessor(LogitsProcessor):
    def __init__(self, cfg_str: str, tokenizer: LlamaCppTokenizer):
        """Compile the FSM that drives the CFG-guided generation.

        Parameters
        ----------
        cfg_str
            A string that represents a grammar
        tokenizer
            An instance of `Tokenizer`

        """
        fsm = CFGFSM(cfg_str, tokenizer)
        super().__init__(tokenizer, fsm)
