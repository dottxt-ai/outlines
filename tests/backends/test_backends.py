import outlines
import pytest
import transformers

from outlines.backends import (
    _get_backend,
    get_json_schema_logits_processor,
    get_regex_logits_processor,
    get_cfg_logits_processor,
)
from outlines.backends.outlines_core import (
    OutlinesCoreBackend,
    OutlinesCoreLogitsProcessor,
)
from outlines.backends.llguidance import (
    LLGuidanceBackend,
    LLGuidanceLogitsProcessor
)
from outlines.backends.xgrammar import XGrammarBackend, XGrammarLogitsProcessor


@pytest.fixture
def model():
    return outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )

@pytest.fixture
def json_schema():
    return (
        '{"type": "object", "properties": {"name": {"type": "string"}, '
        + '"age": {"type": "integer"}}, "required": ["name", "age"], '
        + '"additionalProperties": false}'
    )

@pytest.fixture
def regex():
    return r"[0-9]{3}"

@pytest.fixture
def cfg_lark():
    return """
?start: sum

?sum: product
| sum "+" product   -> add
| sum "-" product   -> sub

?product: atom
| product "*" atom  -> mul
| product "/" atom  -> div

?atom: NUMBER           -> number
| "-" atom         -> neg
| "(" sum ")"

%import common.NUMBER
%import common.WS_INLINE

%ignore WS_INLINE
"""

@pytest.fixture
def cfg_ebnf():
    return """
root ::= answer
answer ::= "yes" | "no"
"""


def test_get_backend(model):
    backend = _get_backend("outlines_core", model)
    assert isinstance(backend, OutlinesCoreBackend)

    backend = _get_backend("xgrammar", model)
    assert isinstance(backend, XGrammarBackend)

    backend = _get_backend("llguidance", model)
    assert isinstance(backend, LLGuidanceBackend)

    with pytest.raises(ValueError, match="not supported"):
        _get_backend("not_supported", model)


def test_get_json_schema_logits_processor(model, json_schema):
    processor = get_json_schema_logits_processor("outlines_core", model, json_schema)
    assert isinstance(processor, OutlinesCoreLogitsProcessor)

    processor = get_json_schema_logits_processor("llguidance", model, json_schema)
    assert isinstance(processor, LLGuidanceLogitsProcessor)

    processor = get_json_schema_logits_processor("xgrammar", model, json_schema)
    assert isinstance(processor, XGrammarLogitsProcessor)


def test_get_regex_logits_processor(model, regex):
    processor = get_regex_logits_processor("outlines_core", model, regex)
    assert isinstance(processor, OutlinesCoreLogitsProcessor)

    processor = get_regex_logits_processor("llguidance", model, regex)
    assert isinstance(processor, LLGuidanceLogitsProcessor)

    processor = get_regex_logits_processor("xgrammar", model, regex)
    assert isinstance(processor, XGrammarLogitsProcessor)


def test_get_cfg_logits_processor(model, cfg_lark, cfg_ebnf):
    with pytest.raises(
        NotImplementedError,
        match="Outlines Core does not support context-free grammar."
    ):
        get_cfg_logits_processor("outlines_core", model, cfg_lark)

    processor = get_cfg_logits_processor("llguidance", model, cfg_lark)
    assert isinstance(processor, LLGuidanceLogitsProcessor)

    processor = get_cfg_logits_processor("xgrammar", model, cfg_ebnf)
    assert isinstance(processor, XGrammarLogitsProcessor)
