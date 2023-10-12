import torch

from outlines.text.generate.continuation import Continuation, continuation


class Tokenizer:
    eos_token = "<EOS>"
    eos_token_id = 0
    pad_token_id = -1

    def decode(self, token_ids):
        return ["Test"] * token_ids.shape[0]


class Model:
    tokenizer = Tokenizer()
    device = "cpu"


def test_continuation_eos_is_finished():
    model = continuation(Model())
    assert isinstance(model, Continuation)

    token_ids = torch.tensor([[3, 2]])
    result = model.is_finished(token_ids)
    assert torch.equal(result, torch.tensor([False]))

    token_ids = torch.tensor([[3, 2, 0]])
    result = model.is_finished(token_ids)
    assert torch.equal(result, torch.tensor([True]))

    token_ids = torch.tensor([[3, 2, 1], [3, 2, 0]])
    result = model.is_finished(token_ids)
    assert torch.equal(result, torch.tensor([False, True]))

    token_ids = torch.tensor([[3, 2, 1, 0], [3, 2, 0, -1]])
    result = model.is_finished(token_ids)
    assert torch.equal(result, torch.tensor([True, False]))


def test_continuation_postprocess():
    model = continuation(Model())
    result = model.postprocess_completions(["Here<EOS>"])
    assert len(result) == 1
    assert result[0] == "Here"


def test_continuation_stop_is_finished():
    tokenizer = Tokenizer()
    tokenizer.decode = lambda x: ["finished \n", "not_finished"]
    model = Model()
    model.tokenizer = tokenizer

    model = continuation(model, stop=["\n"])

    token_ids = torch.tensor([[2, 3], [2, 3]])
    result = model.is_finished(token_ids)
    assert torch.equal(result, torch.tensor([True, False]))


def test_continuation_stop_postprocess():
    model = Continuation(Model(), stop="\n")
    result = model.postprocess_completions(["Stop\n"])
    assert len(result) == 1
    assert result[0] == "Stop"

    model = Continuation(Model(), stop=["\n", ","])
    result = model.postprocess_completions(["Stop"])
    assert len(result) == 1
    assert result[0] == "Stop"

    result = model.postprocess_completions(["Stop\n"])
    assert len(result) == 1
    assert result[0] == "Stop"

    result = model.postprocess_completions(["Stop\naaa"])
    assert len(result) == 1
    assert result[0] == "Stop"

    result = model.postprocess_completions(["Stop,aa\naaa"])
    assert len(result) == 1
    assert result[0] == "Stop"

    result = model.postprocess_completions(["Stop\naa,a"])
    assert len(result) == 1
    assert result[0] == "Stop"

    result = model.postprocess_completions(["Stop\n", "Nonstop"])
    assert len(result) == 2
    assert result == ["Stop", "Nonstop"]

    result = model.postprocess_completions(["StopHere\nNoHere<EOS>"])
    assert len(result) == 1
    assert result[0] == "StopHere"
