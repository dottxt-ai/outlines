"""Utility functions used in integrations with other packages.

 _______________________________
/ Don't want to self-host?       \
\\ Try .json at http://dottxt.co /
 -------------------------------
       \\   ^__^
        \\  (oo)\\_______
            (__)\\       )\\/\
                ||----w |
                ||     ||

Copyright 2024- the Outlines developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
from typing import Type, Union

from pydantic import BaseModel
from transformers import SPIECE_UNDERLINE, PreTrainedTokenizerBase


def adapt_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Adapt a tokenizer to use to compile the FSM.

    The API of Outlines tokenizers is slightly different to that of `transformers`. In
    addition we need to handle the missing spaces to Llama's tokenizer to be able to
    compile FSMs for this model.

    Parameters
    ----------
    tokenizer
        The tokenizer of the model.

    Returns
    -------
    PreTrainedTokenizerBase
        The adapted tokenizer.
    """
    tokenizer.vocabulary = tokenizer.get_vocab()
    tokenizer.special_tokens = set(tokenizer.all_special_tokens)

    def convert_token_to_string(token: Union[str, bytes]) -> str:
        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if (
            type(token) is str
            and token.startswith(SPIECE_UNDERLINE)
            or token == "<0x20>"
        ):
            return " " + string

        return string

    tokenizer.convert_token_to_string = convert_token_to_string

    return tokenizer
