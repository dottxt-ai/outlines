from abc import abstractmethod
from typing import List, Protocol, Union

import numpy as np
import torch
from numpy.typing import NDArray

from outlines.models.tokenizer import Tokenizer


class LogitsProcessor(Protocol):
    tokenizer: Tokenizer

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __copy__(self):
        return self

    @abstractmethod
    def process_logits(
        self, input_ids: List[int], logits: torch.Tensor
    ) -> torch.Tensor:
        ...

    def __call__(
        self,
        input_ids: Union[NDArray[np.int64], List[int], torch.Tensor],
        logits: Union[NDArray[np.float32], torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply logits processor

        Unify type
        - convert input_ids: either ndarray, List[int], or Tensor -> List[int]
        - convert logits: either ndarray or Tensor -> Tensor

        Call process_logits() to perform business logic
        """
        if not isinstance(input_ids, list):
            input_ids = input_ids.tolist()

        if isinstance(logits, np.ndarray):
            # Unify type, convert numpy array to Tensor
            # from_numpy and .numpy() don't copy the data, it uses the same memory address
            torch_logits = torch.from_numpy(logits)
            processed_torch_logits = self.process_logits(input_ids, torch_logits)
            return processed_torch_logits.detach().numpy()
        else:
            return self.process_logits(input_ids, logits)
