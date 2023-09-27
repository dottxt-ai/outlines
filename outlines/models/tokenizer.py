from abc import abstractmethod
from typing import Dict, Hashable, List, Protocol, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class Tokenizer(Protocol, Hashable):
    eos_token: str
    eos_token_id: int
    pad_token_id: int
    vocabulary: Dict[str, int]
    special_tokens: Set[int]

    @abstractmethod
    def encode(
        self, prompt: Union[str, List[str]]
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Translate the input prompts into NumPy arrays of token ids and attention mask."""
        ...

    @abstractmethod
    def decode(self, token_ids: NDArray[np.int64]) -> List[str]:
        """Translate an array of token ids to a string or list of strings."""
        ...

    @abstractmethod
    def convert_token_to_string(self, token: str) -> str:
        """Convert a token to its equivalent string.

        This is for instance useful for BPE tokenizers where whitespaces are
        represented by the special characted `Ġ`. This prevents matching a raw
        token that includes `Ġ` with a string.

        """
        ...
