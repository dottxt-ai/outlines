import importlib

import pytest
import torch

from outlines.generate.generator import sequence_generator, token_generator


@pytest.mark.parametrize("enable_logger", [True, False])
def test_token_generator_logger_call(enable_logger, mocker):
    """log_logits() is expensive, assert only called when explicitly enabled"""

    class MockFSM:
        def next_state(self, state, next_token_ids):
            return 0

        def allowed_token_ids(self, _):
            return []

        def is_final_state(self, _):
            return True

    class MockTokenizer:
        eos_token_id = 2

        def decode(self, _):
            return "x"

    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return torch.argmax(biased_logits, keepdims=True)

    # reset logger state
    import outlines.logging  # type: ignore

    importlib.reload(outlines.logging)
    if enable_logger:
        outlines.logging.enable_logits_logging()

    mock_logits_logger_info = mocker.patch("outlines.logging.logits_logger.info")

    token_ids, attention_mask = (
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]),
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
    )
    init_fsm_states = [0]
    generate = token_generator(MockModel(), sampler)
    sequence = sequence_generator(
        generate,
        [MockFSM()],
        token_ids,
        attention_mask,
        init_fsm_states,
        torch.Generator(),
    )
    next(sequence)

    if enable_logger:
        mock_logits_logger_info.assert_called()
    else:
        mock_logits_logger_info.assert_not_called()

    # ensure enable_logits_logging() doesn't bleed into other tests
    importlib.reload(outlines.logging)
