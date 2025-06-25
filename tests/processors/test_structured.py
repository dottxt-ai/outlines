import pytest
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import outlines
from outlines.processors.guide import RegexGuide
from outlines.processors.structured import (
    GuideLogitsProcessor,
    RegexLogitsProcessor,
    JSONLogitsProcessor,
    CFGLogitsProcessor,
)


arithmetic_grammar = """
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
def tokenizer():
    model = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
        AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )
    return model.tokenizer


def test_structured_init_guide_logits_processor(tokenizer):
    guide = RegexGuide.from_regex(r"[a-z]+", tokenizer)
    processor = GuideLogitsProcessor(
        tokenizer=tokenizer,
        guide=guide,
        tensor_library_name="torch",
    )
    assert isinstance(processor, GuideLogitsProcessor)
    assert processor.tokenizer is tokenizer
    assert processor.guide is guide
    assert processor.tensor_adapter.library_name == "torch"


def test_structured_init_guide_logits_processor_copy(tokenizer):
    guide = RegexGuide.from_regex(r"[a-z]+", tokenizer)
    processor = GuideLogitsProcessor(
        tokenizer=tokenizer,
        guide=guide,
        tensor_library_name="torch",
    )
    processor_copy = processor.copy()
    assert processor_copy is not processor
    assert processor_copy.tokenizer is tokenizer
    assert processor_copy.guide is guide
    assert processor_copy.tensor_adapter.library_name == "torch"


def test_structured_guide_logits_processor_call(tokenizer):
    guide = RegexGuide.from_regex(r"[a-z]+", tokenizer)
    processor = GuideLogitsProcessor(
        tokenizer=tokenizer,
        guide=guide,
        tensor_library_name="torch",
    )
    vocab_token_ids = list(tokenizer.vocabulary.values())
    logits = torch.randn(1, len(vocab_token_ids))
    input_ids = torch.randint(0, len(vocab_token_ids), (1, 10))
    output = processor(input_ids, logits)
    assert output.shape == (1, len(vocab_token_ids))


def test_structured_init_json_logits_processor(tokenizer):
    processor = JSONLogitsProcessor(
        schema={"type": "object", "properties": {"name": {"type": "string"}}},
        tokenizer=tokenizer,
        tensor_library_name="torch",
    )
    assert isinstance(processor, JSONLogitsProcessor)
    assert processor.tokenizer is tokenizer
    assert processor.guide is not None
    assert processor.tensor_adapter.library_name == "torch"


def test_structured_init_regex_logits_processor(tokenizer):
    processor = RegexLogitsProcessor(
        regex_string=r"[a-z]+",
        tokenizer=tokenizer,
        tensor_library_name="torch",
    )
    assert isinstance(processor, RegexLogitsProcessor)
    assert processor.tokenizer is tokenizer
    assert processor.guide is not None
    assert processor.tensor_adapter.library_name == "torch"


def test_structured_init_cfg_logits_processor(tokenizer):
    processor = CFGLogitsProcessor(
        cfg_str=arithmetic_grammar,
        tokenizer=tokenizer,
        tensor_library_name="torch",
    )
    assert isinstance(processor, CFGLogitsProcessor)
    assert processor.tokenizer is tokenizer
    assert processor.guide is not None
    assert processor.tensor_adapter.library_name == "torch"


def test_structured_cfg_logits_processor_call(tokenizer):
    processor = CFGLogitsProcessor(
        cfg_str=arithmetic_grammar,
        tokenizer=tokenizer,
        tensor_library_name="torch",
    )
    vocab_token_ids = list(tokenizer.vocabulary.values())
    logits = torch.randn(1, len(vocab_token_ids))
    input_ids = torch.randint(0, len(vocab_token_ids), (1, 10))
    output = processor(input_ids, logits)
    assert output.shape == (1, len(vocab_token_ids))
    processor(input_ids, logits)
