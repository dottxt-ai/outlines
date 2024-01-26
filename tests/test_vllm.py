import pytest
import torch
from transformers import AutoTokenizer

from outlines.serve.vllm import (
    CFGLogitsProcessor,
    JSONLogitsProcessor,
    RegexLogitsProcessor,
)

TEST_REGEX = r"(-)?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-][0-9]+)?"
TEST_CFG = """
start: DECIMAL
DIGIT: "0".."9"
INT: DIGIT+
DECIMAL: INT "." INT? | "." INT
"""
TEST_SCHEMA = '{"type": "string", "maxLength": 5}'

LOGIT_PROCESSORS = (
    (CFGLogitsProcessor, TEST_CFG),
    (RegexLogitsProcessor, TEST_REGEX),
    (JSONLogitsProcessor, TEST_SCHEMA),
)

TEST_MODEL = "hf-internal-testing/tiny-random-GPTJForCausalLM"


@pytest.mark.parametrize("logit_processor, fsm_str", LOGIT_PROCESSORS)
def test_logit_processor(logit_processor, fsm_str: str):
    class MockvLLMEngine:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.float), None

    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    engine = MockvLLMEngine(tokenizer)
    logit_processor(fsm_str, engine)
    assert isinstance(engine.tokenizer.decode([0, 1, 2, 3]), list)
    logit_processor(fsm_str, engine)
    assert isinstance(engine.tokenizer.decode([0, 1, 2, 3]), list)
