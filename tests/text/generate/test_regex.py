import math

import interegular
import pytest
import torch

import outlines.text.generate as generate
from outlines.text.fsm import create_fsm_index_tokenizer, make_deterministic_fsm
from outlines.text.generate.regex import Regex


class Tokenizer:
    eos_token = "<EOS>"
    pad_token = None
    eos_token_id = 0
    pad_token_id = -1
    vocabulary = {"<EOS>": 0, "-": 1, "1": 2, "0.": 3, "431": 4, "a": 5, "A": 6}
    tokens = list(vocabulary.keys())
    special_tokens = {"<EOS>"}

    def encode(self, tokens):
        if not isinstance(tokens, (tuple, list)):
            tokens = [tokens]

        return [self.vocabulary[token] for token in tokens]

    def decode(self, token_ids):
        decoded = []
        for i in range(token_ids.shape[0]):
            decoded.append("".join([self.tokens[idx] for idx in token_ids[i]]))

        return decoded

    def convert_token_to_string(self, token):
        return token


class TokenizerWithEmpty(Tokenizer):
    vocabulary = {"<EOS>": 0, "-": 1, "1": 2, "0.": 3, "431": 4, "a": 5, "A": 6, "": 7}
    tokens = list(vocabulary.keys())


class Model:
    tokenizer = Tokenizer()
    device = "cpu"


class ModelWithEmpty:
    tokenizer = TokenizerWithEmpty()
    device = "cpu"


@pytest.mark.parametrize(
    "regex_string, valid_first_token, proposal",
    [
        (
            r"[A-Z]+",
            6,
            [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 1.0],
        ),
        (
            r"[a-z]+",
            5,
            [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 1.0, -math.inf],
        ),
        (
            r"(a|A)",
            6,
            [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 1.0, 1.0],
        ),
        (r"\d+", 2, [-math.inf, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf]),
        (r"\d+\.", 3, [-math.inf, -math.inf, 1.0, 1.0, 1.0, -math.inf, -math.inf]),
    ],
)
def test_regex_proposal(regex_string, valid_first_token, proposal):
    model = Model()
    generator = generate.regex(model, regex_string)

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor([[]]), logits)
    assert torch.equal(result.squeeze(), torch.tensor(proposal))
    assert result.squeeze()[0] == -math.inf

    # The EOS token can be generated once the FSM is in an accept state
    result = generator.create_proposal(torch.tensor([[valid_first_token]]), logits)
    assert result.squeeze()[0] == 1


def test_regex_no_valid_transition():
    model = Model()
    with pytest.raises(ValueError, match="The vocabulary does not allow"):
        generate.regex(model, "aw")


@pytest.mark.parametrize(
    "input_ids, proposal",
    [
        ([[]], [[-math.inf, 1.0, 1.0, -math.inf, 1.0, -math.inf, -math.inf]]),
        ([[1]], [[-math.inf, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf]]),
        ([[4]], [[1.0, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf]]),
        (
            [[4], [2]],
            [
                [1.0, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf],
                [1.0, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf],
            ],
        ),
        (
            [[4, 0], [1, 2]],
            [
                [1.0, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                [1.0, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf],
            ],
        ),
    ],
)
def test_integer_proposal(input_ids, proposal):
    model = Model()
    generator = generate.integer(model)

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor(input_ids), logits)
    assert torch.equal(
        result,
        torch.tensor(proposal),
    )


def test_choice_proposal():
    model = Model()
    generator = generate.choice(model, ["1", "431a", "431A-"])
    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor([[]]), logits)
    assert torch.equal(
        result,
        torch.tensor(
            [[-math.inf, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf]]
        ),
    )

    result = generator.create_proposal(torch.tensor([[4]]), logits)
    assert torch.equal(
        result,
        torch.tensor(
            [[-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 1.0, 1.0]]
        ),
    )

    result = generator.create_proposal(torch.tensor([[4, 6]]), logits)
    assert torch.equal(
        result,
        torch.tensor(
            [[-math.inf, 1.0, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf]]
        ),
    )


@pytest.mark.parametrize(
    "input_ids, proposal",
    [
        ([[]], [[-math.inf, 1.0, 1.0, 1.0, 1.0, -math.inf, -math.inf]]),
        ([[3]], [[1.0, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf]]),
    ],
)
def test_float_proposal(input_ids, proposal):
    model = Model()
    generator = generate.float(model)

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor(input_ids), logits)
    assert torch.equal(
        result,
        torch.tensor(proposal),
    )


@pytest.mark.parametrize(
    "input_ids, proposal, with_empty",
    [
        ([[]], [[-math.inf, 1.0, 1.0, 1.0, 1.0, -math.inf, -math.inf, 1]], True),
        (
            [[]],
            [[-math.inf, 1.0, 1.0, 1.0, 1.0, -math.inf, -math.inf, -math.inf]],
            False,
        ),
        ([[3]], [[1.0, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf, 1]], True),
        (
            [[3]],
            [[1.0, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf, -math.inf]],
            False,
        ),
    ],
)
def test_empty_strings(input_ids, proposal, with_empty):
    model = ModelWithEmpty()
    generator = generate.float(model, allow_empty_tokens=with_empty)

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor(input_ids), logits)
    assert torch.equal(
        result,
        torch.tensor(proposal),
    )


def test_Regex_precomputed():
    model = Model()
    choices = ["1", "431a", "431A-"]
    regex_str = r"(" + r"|".join(choices) + r")"

    regex_pattern = interegular.parse_pattern(regex_str)
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

    (
        states_to_token_maps,
        empty_token_ids,
    ) = create_fsm_index_tokenizer(regex_fsm, model.tokenizer)

    generator = Regex(
        model,
        regex_str,
        max_tokens=100,
        initial_state=regex_fsm.initial,
        final_states=regex_fsm.finals,
        states_to_token_maps=states_to_token_maps,
        empty_token_ids=empty_token_ids,
    )

    logits = torch.ones(len(model.tokenizer.vocabulary))
    result = generator.create_proposal(torch.tensor([[]]), logits)
    assert torch.equal(
        result,
        torch.tensor(
            [[-math.inf, -math.inf, 1.0, -math.inf, 1.0, -math.inf, -math.inf]]
        ),
    )
