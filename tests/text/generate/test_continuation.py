import numpy as np
from numpy.testing import assert_array_equal

from outlines.text.generate.continuation import Continuation, continuation


class Tokenizer:
    eos_token = "<EOS>"
    eos_token_id = 0
    pad_token_ids = -1


class Model:
    tokenizer = Tokenizer()


def test_continuation_is_finished():
    model = continuation(Model(), 10)
    assert isinstance(model, Continuation)

    token_ids = np.array([[3, 2]])
    result = model.is_finished(token_ids)
    assert_array_equal(result, [False])

    token_ids = np.array([[3, 2, 0]])
    result = model.is_finished(token_ids)
    assert_array_equal(result, [True])

    token_ids = np.array([[3, 2, 1], [3, 2, 0]])
    result = model.is_finished(token_ids)
    assert_array_equal(result, [False, True])

    token_ids = np.array([[3, 2, 1, 0], [3, 2, 0, -1]])
    result = model.is_finished(token_ids)
    assert_array_equal(result, [True, False])


def test_continuation_postprocess():
    model = continuation(Model())
    result = model.postprocess_completions(["Here<EOS>"])
    assert len(result) == 1
    assert result[0] == "Here"
