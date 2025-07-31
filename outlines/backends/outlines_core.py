"""Backend class for Outlines Core."""

from typing import Dict, Optional

from outlines_core import Guide, Index, Vocabulary
# TODO: change this once the import issue is fixed in outlines_core
from outlines_core import outlines_core

from outlines.backends.base import BaseBackend
from outlines.models import SteerableModel
from outlines.models.llamacpp import LlamaCpp
from outlines.models.mlxlm import MLXLM
from outlines.models.transformers import Transformers
from outlines.processors.base_logits_processor import (
    OutlinesLogitsProcessor,
    TensorType
)


class OutlinesCoreLogitsProcessor(OutlinesLogitsProcessor):
    """Logits processor for Outlines Core."""

    def __init__(
        self,
        index: Index,
        tensor_library_name: str,
        end_thinking_token_id: Optional[int] = None
    ):
        """
        Parameters
        ----------
        index: Index
            The Outlines Core `Index` instance to use to create the Outlines
            Core `Guide` instances that will be used to bias the logits
        tensor_library_name: str
            The tensor library name to use for the logits processor.

        """
        self.index = index
        self.tensor_library_name = tensor_library_name
        self.first_token = True
        self.end_thinking_token_id = end_thinking_token_id
        super().__init__(tensor_library_name)

    def reset(self) -> None:
        """Ensure self._setup is called again for the next generation."""
        self.first_token = True

    def _setup(self, batch_size: int, vocab_size: int) -> None:
        """Set the guides, bitmasks and some functions used in the
        `process_logits` method.

        This method is called when the first token is generated instead of
        at initialization because we need to know the batch size.

        """
        self._is_thinking = [self.end_thinking_token_id is not None] * batch_size

        if self.tensor_library_name == "torch":
            from outlines_core.kernels.torch import allocate_token_bitmask

            self.allocate_token_bitmask = allocate_token_bitmask
            self.bias_logits = self._bias_logits_torch

        elif self.tensor_library_name == "numpy":
            from outlines_core.kernels.numpy import allocate_token_bitmask

            self.allocate_token_bitmask = allocate_token_bitmask
            self.bias_logits = self._bias_logits_numpy

        elif self.tensor_library_name == "mlx":
            from outlines_core.kernels.mlx import (
                allocate_token_bitmask
            )

            self.allocate_token_bitmask = allocate_token_bitmask
            self.bias_logits = self._bias_logits_mlx

        else:
            raise ValueError(
                f"Unsupported tensor library: {self.tensor_library_name}"
            )

        self._guides = [Guide(self.index) for _ in range(batch_size)]
        
        # Save the guide content to a file
        with open('guide_debug.txt', 'w') as f:
            f.write(str(self._guides))

        self._bitmasks = [
            self.allocate_token_bitmask(vocab_size)
            for _ in range(batch_size)
        ]

    def _bias_logits_mlx( # pragma: no cover
        self, batch_size: int, logits: TensorType
    ) -> TensorType:
        """Bias the logits for MLX tensors."""
        from outlines_core.kernels.mlx import (
            apply_token_bitmask,
            fill_next_token_bitmask
        )

        biased_logits_array = []
        for i in range(batch_size):
            fill_next_token_bitmask(self._guides[i], self._bitmasks[i])
            biased_logits = apply_token_bitmask(
                self.tensor_adapter.unsqueeze(logits[i]), self._bitmasks[i] # type: ignore
            )
            biased_logits_array.append(biased_logits)

        return self.tensor_adapter.concatenate(biased_logits_array)

    def _bias_logits_torch(
        self, batch_size: int, logits: TensorType
    ) -> TensorType:
        """Bias the logits for Torch tensors."""
        from outlines_core.kernels.torch import (
            apply_token_bitmask_inplace,
            fill_next_token_bitmask
        )

        for i in range(batch_size):
            if not self._is_thinking[i]:
                fill_next_token_bitmask(self._guides[i], self._bitmasks[i])
                apply_token_bitmask_inplace(
                    self.tensor_adapter.unsqueeze(logits[i]), # type: ignore
                    self._bitmasks[i]
                )

        return logits

    def _bias_logits_numpy(
        self, batch_size: int, logits: TensorType
    ) -> TensorType:
        """Bias the logits for Numpy tensors."""
        from outlines_core.kernels.numpy import (
            apply_token_bitmask_inplace,
            fill_next_token_bitmask
        )

        for i in range(batch_size):
            fill_next_token_bitmask(self._guides[i], self._bitmasks[i])
            apply_token_bitmask_inplace(
                self.tensor_adapter.unsqueeze(logits[i]), # type: ignore
                self._bitmasks[i]
            )

        return logits

    def process_logits(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Use the guides to bias the logits.

        Parameters
        ----------
        input_ids
            The ids of the tokens of the existing sequences.
        logits
            The logits for the current generation step.

        Returns
        -------
        TensorType
            The biased logits.

        """
        batch_size = self.tensor_adapter.shape(input_ids)[0]
        vocab_size = self.tensor_adapter.shape(logits)[1]

        if self.first_token:
            self._setup(batch_size, vocab_size)
            self.first_token = False
        else:
            for i in range(batch_size):
                latest_token_id = input_ids[i][-1] # type: ignore
                if not self._is_thinking[i]:
                    self._guides[i].advance(
                        token_id=latest_token_id,
                        return_tokens=False
                    )
                else:
                    if latest_token_id == self.end_thinking_token_id:
                        self._is_thinking[i] = False

        return self.bias_logits(batch_size, logits)


class OutlinesCoreBackend(BaseBackend):
    """Backend for Outlines Core."""

    def __init__(self, model: SteerableModel, end_thinking_tag: Optional[str] = None):
        """
        Parameters
        ----------
        model
            The Outlines model of the user.
        end_thinking_tag
            The tag to use to identify the end of thinking.

        """
        if isinstance(model, Transformers):
            tokenizer = model.hf_tokenizer
            vocabulary = tokenizer.get_vocab()
            eos_token_id = tokenizer.eos_token_id
            eos_token = tokenizer.eos_token
        elif isinstance(model, LlamaCpp):
            tokenizer = model.tokenizer
            vocabulary = tokenizer.vocabulary
            eos_token_id = tokenizer.eos_token_id
            eos_token = tokenizer.eos_token
        elif isinstance(model, MLXLM):
            tokenizer = model.mlx_tokenizer
            vocabulary = tokenizer.get_vocab()
            eos_token_id = tokenizer.eos_token_id
            eos_token = tokenizer.eos_token
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        
        print("eos_token_id", eos_token_id)

        self.eos_token_id = eos_token_id
        self.vocabulary = self.create_outlines_core_vocabulary(
            vocabulary, eos_token_id, eos_token
        )
        self.tensor_library_name = model.tensor_library_name
        self.end_thinking_token_id = tokenizer.encode(end_thinking_tag)[0] if end_thinking_tag else None

    def get_json_schema_logits_processor(
        self, json_schema: str
    ):
        """Create a logits processor from a JSON schema.

        Parameters
        ----------
        json_schema: str
            The JSON schema to create a logits processor from.

        Returns
        -------
        LogitsProcessor
            The logits processor to use to constrain the generation.

        """
        regex = outlines_core.json_schema.build_regex_from_schema(json_schema)
        return self.get_regex_logits_processor(regex)

    def get_regex_logits_processor(self, regex: str):
        """Create a logits processor from a regex.

        Parameters
        ----------
        regex: str
            The regex to create a logits processor from.

        Returns
        -------
        LogitsProcessor
            The logits processor to use to constrain the generation.

        """
        index = Index(regex, self.vocabulary)
        print("hereeeee", regex)
        return OutlinesCoreLogitsProcessor(index, self.tensor_library_name, self.end_thinking_token_id)

    def get_cfg_logits_processor(self, grammar):
        raise NotImplementedError(
            "Outlines Core does not support context-free grammar."
        )

    @staticmethod
    def create_outlines_core_vocabulary(
        vocab: Dict[str, int], eos_token_id: int, eos_token: str
    ) -> Vocabulary:
        """Create an Outlines Core Vocabulary instance.

        Parameters
        ----------
        vocab: Dict[str, int]
            The vocabulary to create an Outlines Core vocabulary from.
        eos_token_id: int
            The EOS token ID.
        eos_token: str
            The EOS token.

        Returns
        -------
        Vocabulary
            The Outlines Core Vocabulary instance.

        """
        formatted_vocab = {}
        for token, token_id in vocab.items():
            formatted_vocab[token] = [token_id]
        formatted_vocab.pop(eos_token)
        return Vocabulary(eos_token_id, formatted_vocab)
