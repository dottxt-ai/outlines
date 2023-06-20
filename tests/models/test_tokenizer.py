import pytest

from outlines.models.tokenizer import Tokenizer


def test_tokenizer():
    with pytest.raises(TypeError, match="instantiate abstract"):
        Tokenizer()
