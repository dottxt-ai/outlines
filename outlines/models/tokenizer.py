from typing import TYPE_CHECKING, Dict, Hashable, List, Protocol, Set, Tuple, Union

if TYPE_CHECKING:
    import torch


class Tokenizer(Hashable, Protocol):
    eos_token: str
    eos_token_id: int
    pad_token_id: int
    vocabulary: Dict[str, int]
    special_tokens: Set[str]

    def encode(
        self, prompt: Union[str, List[str]]
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Translate the input prompts into 2-dimensional tensors of token ids and attention mask."""
        ...

    def decode(self, token_ids: List[List[int]]) -> List[str]:
        """Translate a list of lists of token ids into a list of strings."""
        ...

    def convert_token_to_string(self, token: str) -> str:
        """Convert a token to its equivalent string.

        This is for instance useful for BPE tokenizers where whitespaces are
        represented by the special characted `Ġ`. This prevents matching a raw
        token that includes `Ġ` with a string.
        """
        ...
