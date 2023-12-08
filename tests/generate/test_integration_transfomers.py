import datetime
import re
from enum import Enum
from typing import List, Union

import pytest
import torch
from pydantic import BaseModel, constr

import outlines.generate as generate
import outlines.models as models
from outlines.fsm.regex import reduced_vocabulary
from outlines.models.transformers import TransformerTokenizer


def test_deprecation():
    import outlines

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")

    with pytest.warns(DeprecationWarning):
        outlines.text.generate.continuation(model, max_tokens=10)

        with pytest.raises(NotImplementedError):
            outlines.text.generate.continuation(model, max_tokens=10, stop="string")

    with pytest.warns(DeprecationWarning):
        outlines.text.generate.choice(model, ["A", "B"], max_tokens=10)

    with pytest.warns(DeprecationWarning):
        outlines.text.generate.regex(model, "[0-9]", max_tokens=10)

    with pytest.warns(DeprecationWarning):
        outlines.text.generate.format(model, int, max_tokens=10)

    with pytest.warns(DeprecationWarning):

        def function(a: int):
            pass

        outlines.text.generate.json(model, function, max_tokens=10)


def test_transformers_integration_text():
    rng = torch.Generator()
    rng.manual_seed(10000)  # Choosen so <EOS> is generated

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    sequence = generate.text(model)("Write a short sentence ", rng=rng)
    assert isinstance(sequence, str)
    assert model.tokenizer.eos_token not in sequence

    sequence = generate.text(model, max_tokens=10)("Write a short sentence ", rng=rng)
    assert isinstance(sequence, str)

    prompts = ["Write a short sentence ", "And another one "]
    sequence = generate.text(model, max_tokens=10)(prompts, rng=rng)
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    assert isinstance(sequence[0], str)


def test_transformers_integration_streaming():
    rng = torch.Generator()
    rng.manual_seed(10000)  # Choosen so <EOS> is generated

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    sequence = generate.text(model, max_tokens=10).stream(
        "Write a short sentence ", rng=rng
    )

    token = next(sequence)
    assert isinstance(token, list)
    assert isinstance(token[0], str)

    remaining = "".join([token[0] for token in sequence])
    assert isinstance(remaining, str)

    sequence = generate.text(model, max_tokens=10).stream(
        ["Prompt1", "Prompt2"], rng=rng
    )
    tokens = next(sequence)
    assert isinstance(tokens, list)
    assert isinstance(tokens[0], str)
    assert isinstance(tokens[1], str)


@pytest.mark.xfail(reason="not implemented")
def test_transformers_integration_text_stop():
    rng = torch.Generator()
    rng.manual_seed(10000)  # Choosen so <EOS> is generated

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompt = "Write a short sentence "
    sequence = generate.text(model, stop="a")(prompt, rng=rng)
    assert sequence[len(prompt) :].find("a") == -1


@pytest.mark.xfail(reason="not implemented")
def test_transformers_integration_text_array_samples():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompts = ["Write a short sentence", "And another one"]
    _ = generate.text(model, max_tokens=10)(prompts, rng=rng, samples=3)


def test_transformers_various_regexes():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompt = "Write an email address"
    regex_str = r"([a-z]{10})@([a-z]{5})\.([a-z]{3})"
    generator = generate.regex(model, regex_str)

    # One prompt
    sequence = generator(prompt, rng=rng)
    assert re.fullmatch(regex_str, sequence) is not None


def test_transformers_various_regexes_prompt_list():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompt = "Write an email address"
    regex_str = r"([a-z]{10})@([a-z]{5})\.([a-z]{3})"
    generator = generate.regex(model, regex_str)

    # Two prompts
    sequence = generator([prompt, prompt], rng=rng)
    assert re.fullmatch(regex_str, sequence[0]) is not None
    assert re.fullmatch(regex_str, sequence[1]) is not None


def test_transformers_integration_integer():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name)
    prompt = "Write a short sentence"
    sequence = generate.format(model, int, max_tokens=10)(prompt, rng=rng)

    assert sequence != ""
    int(sequence)


def test_transformers_integration_integer_array():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name)
    prompts = ["Give me a number", "And another one"]
    sequence = generate.format(model, int, max_tokens=10)(prompts, rng=rng)
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    int(sequence[0])
    int(sequence[1])


def test_transformers_integration_float():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name)
    prompt = "Write a short sentence"
    sequence = generate.format(model, float, max_tokens=10)(prompt, rng=rng)

    assert sequence != ""
    float(sequence)


def test_transformers_integration_bool():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name)
    prompt = "Is this True or False?"
    sequence = generate.format(model, bool, max_tokens=10)(prompt, rng=rng)

    assert sequence != ""
    bool(sequence)


def test_transformers_integration_date():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name)
    prompt = "What day is it today?"
    sequence = generate.format(model, datetime.date, max_tokens=10)(prompt, rng=rng)

    assert sequence != ""
    datetime.datetime.strptime(sequence, "%Y-%m-%d")


def test_transformers_integration_time():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name)
    prompt = "What time is it?"
    sequence = generate.format(model, datetime.time, max_tokens=10)(prompt, rng=rng)

    assert sequence != ""
    datetime.datetime.strptime(sequence, "%H:%M:%S")


def test_transformers_integration_datetime():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name)
    prompt = "What time is it?"
    sequence = generate.format(model, datetime.datetime, max_tokens=20)(prompt, rng=rng)

    assert sequence != 0
    datetime.datetime.strptime(sequence, "%Y-%m-%d %H:%M:%S")


def test_transformers_integration_choice():
    rng = torch.Generator()
    rng.manual_seed(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompt = "Write a short sentence "
    sequence = generate.choice(model, ["test", "choice"])(prompt, rng=rng)

    assert sequence == "test" or sequence == "choice"


def test_transformers_integration_with_pad_token():
    model_name = "hf-internal-testing/tiny-random-XLMRobertaXLForCausalLM"
    model = models.transformers(model_name, device="meta")
    assert model.tokenizer.pad_token_id == 1
    assert model.tokenizer.pad_token == "<pad>"


def test_transformers_json_basic():
    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompt = "Output some JSON "

    class Spam(BaseModel):
        foo: int
        bar: float
        spam: constr(max_length=10)
        fuzz: bool

    rng = torch.Generator()
    rng.manual_seed(0)  # make sure that `bar` is not an int

    result = generate.json(model, Spam, max_tokens=500)(prompt, rng=rng)
    assert isinstance(result, BaseModel)
    assert isinstance(result.foo, int)
    assert isinstance(result.bar, float)
    assert isinstance(result.spam, str)
    assert isinstance(result.fuzz, bool)
    assert len(result.spam) <= 10


def test_transformers_json_batch():
    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompts = ["Output some JSON ", "Output more JSON"]

    class Spam(BaseModel):
        foo: int
        bar: float
        spam: constr(max_length=10)
        fuzz: bool

    rng = torch.Generator()
    rng.manual_seed(0)  # make sure that `bar` is not an int

    result = generate.json(model, Spam, max_tokens=500)(prompts, rng=rng)
    assert isinstance(result[0], BaseModel)
    assert isinstance(result[1], BaseModel)


def test_transformers_json_str_enum():
    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompt = "Output some JSON "

    rng = torch.Generator()
    rng.manual_seed(0)

    class Name(str, Enum):
        john = "John"
        marc = "Marc"
        michel = "Michel"

    class User(BaseModel):
        user_id: int
        name: Name

    result = generate.json(model, User)(prompt, rng=rng)
    assert isinstance(result, BaseModel)
    assert isinstance(result.user_id, int)
    assert result.name in ["John", "Marc", "Michel"]


def test_transformers_json_int_enum():
    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompt = "Output some JSON "

    rng = torch.Generator()
    rng.manual_seed(0)

    class Id(int, Enum):
        one = 1
        two = 2

    class User(BaseModel):
        user_id: Id

    result = generate.json(model, User)(prompt, rng=rng)
    assert isinstance(result, BaseModel)
    assert isinstance(result.user_id, int)
    assert result.user_id in [1, 2]


def test_transformers_json_array():
    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompt = "Output some JSON "

    class User(BaseModel):
        user_id: int
        value: List[float]

    rng = torch.Generator()
    rng.manual_seed(0)

    result = generate.json(model, User)(prompt, rng=rng)
    assert isinstance(result, BaseModel)
    assert isinstance(result.user_id, int)
    assert isinstance(result.value, list)
    for value in result.value:
        assert isinstance(value, float) or isinstance(value, int)


@pytest.mark.xfail(reason="The implementation of `anyOf` is incorrect")
def test_transformers_json_union():
    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    prompt = "Output some JSON "

    class Spam(BaseModel):
        foo: int
        bar: Union[constr(max_length=10), float]

    rng = torch.Generator()
    rng.manual_seed(4)

    result = generate.json(model, Spam, max_tokens=100)(prompt, rng=rng)
    assert isinstance(result, BaseModel)
    assert (
        isinstance(result.bar, int)
        or isinstance(result.bar, float)
        or isinstance(result.bar, str)
    )


def test_transformers_json_function():
    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name)
    prompt = "Output arguments for the function"

    def function(foo: int, bar: List[int]):
        return foo + sum(bar)

    rng = torch.Generator()
    rng.manual_seed(4)

    sequence = generate.json(model, function, max_tokens=100)(prompt, rng=rng)
    assert isinstance(sequence, dict)
    assert isinstance(function(**sequence), int)


def test_transformers_logits_vocab_size():
    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")

    # Artificially increase the weights/logits size relative
    # to the vocabulary
    model.model.resize_token_embeddings(pad_to_multiple_of=3)

    assert len(model.tokenizer.vocabulary) == 1024
    assert model.model.base_model.wte.weight.shape[0] == 1026

    generator = generate.choice(model, ["True", "False"])

    rng = torch.Generator()
    rng.manual_seed(101)

    sequence = generator("blah", rng=rng)
    assert sequence == "False"


def test_transformers_reduced_vocabulary_caching():
    tokenizer = TransformerTokenizer("gpt2")
    tokenizer2 = TransformerTokenizer("gpt2")

    # TODO: We might actually want only one copy of a given tokenizer.
    assert tokenizer is not tokenizer2

    vocab = reduced_vocabulary(tokenizer)
    vocab2 = reduced_vocabulary(tokenizer2)

    assert vocab2 is vocab


def test_custom_sampler():
    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"

    model = models.transformers(model_name)

    seen = False
    target_token_ids = model.tokenizer.encode(["c"])[0]

    def biased_sampler(
        logits: torch.DoubleTensor, samples: int, *_
    ) -> torch.DoubleTensor:
        nonlocal seen

        if not seen:
            seen = True
            return target_token_ids
        else:
            return torch.tensor([[model.tokenizer.eos_token_id]])

    generator = generate.choice(model, ["a", "b", "c"], sampler=biased_sampler)
    sequence = generator(
        """What is 1+1?
    a. 3
    b. 4
    c. 2"""
    )

    assert sequence == "c"
