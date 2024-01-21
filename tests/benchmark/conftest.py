import pytest

from outlines.fsm.fsm import RegexFSM
from outlines.models.transformers import TransformerTokenizer


@pytest.fixture
def tokenizer():
    return TransformerTokenizer("gpt2")


@pytest.fixture
def ensure_numba_compiled(tokenizer):
    RegexFSM("a", tokenizer)
    return True


test_outputs = []


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write("\n")
    terminalreporter.write("Additional Benchmark Details:\n", bold=True)
    for output in test_outputs:
        terminalreporter.write(f"{output}\n")


def add_test_output(data):
    test_outputs.append(data)
