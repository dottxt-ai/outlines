import datetime
import re
from enum import Enum
from typing import List, Union

import pytest
import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, constr

import outlines.generate as generate
from outlines.models import llamacpp

TEST_MODEL = "./llama-test-model/TinyMistral-248M-v2-Instruct.Q4_K_M.gguf"


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    tmp_path_factory.mktemp("./llama-test-model")
    hf_hub_download(
        repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        local_dir="./llama-test-model",
        local_dir_use_symlinks="auto",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
    )
    return llamacpp(model_path=TEST_MODEL, device="cpu")


def test_llamacpp_integration_text(model):
    model.model.reset()
    sequence = generate.text(model)(
        "<|im_start|>user\nWrite a short sentence<|im_end|>\n<|im_start|>assistant\n"
    )
    assert isinstance(sequence, str)

    sequence = generate.text(model)(
        "<|im_start|>user\nWrite a short sentence<|im_end|>\n<|im_start|>assistant\n",
        max_tokens=10,
        stop_at=".",
    )
    assert isinstance(sequence, str)

    prompts = [
        "<|im_start|>user\nWrite a short sentence<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nAnd another one<|im_end|>\n<|im_start|>assistant\n",
    ]
    sequence = generate.text(model)(prompts, max_tokens=10, stop_at=[".", ","])
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    assert isinstance(sequence[0], str)


def test_llamacpp_integration_text_stop(model):
    model.model.reset()
    prompt = (
        "<|im_start|>user\nWrite a short sentence<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.text(model)(prompt, stop_at="a")
    assert isinstance(sequence, str)
    assert sequence[len(prompt) :].find("a") == -1


def test_llamacpp_various_regexes(model):
    model.model.reset()
    prompt = (
        "<|im_start|>user\nWrite an email address<|im_end|>\n<|im_start|>assistant\n"
    )
    regex_str = r"([a-z]{10})@([a-z]{5})\.([a-z]{3})"
    generator = generate.regex(model, regex_str)

    # One prompt
    sequence = generator(prompts=prompt)
    assert isinstance(sequence, str)
    assert re.fullmatch(pattern=regex_str, string=sequence) is not None


def test_llamacpp_various_regexes_prompt_list(model):
    model.model.reset()
    prompt = (
        "<|im_start|>user\nWrite an email address<|im_end|>\n<|im_start|>assistant\n"
    )
    regex_str = r"([a-z]{10})@([a-z]{5})\.([a-z]{3})"
    generator = generate.regex(model, regex_str)

    # Two prompts
    sequence = generator(prompts=[prompt, prompt])
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    for s in sequence:
        assert isinstance(s, str)
        assert re.fullmatch(pattern=regex_str, string=s) is not None


def test_llamacpp_integration_integer(model):
    model.model.reset()
    prompt = (
        "<|im_start|>user\nWrite a short sentence<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.format(model, int)(prompt, max_tokens=10)
    assert isinstance(sequence, int)
    assert sequence != ""
    int(sequence)


def test_llamacpp_integration_integer_array(model):
    model.model.reset()
    prompts = ["Give me a number", "And another one"]
    sequence = generate.format(model, int)(prompts, max_tokens=10)

    assert isinstance(sequence, list)
    assert len(sequence) == 2
    for s in sequence:
        assert isinstance(s, int)
        int(s)


def test_llamacpp_integration_float(model):
    model.model.reset()
    prompt = (
        "<|im_start|>user\nWrite a short sentence<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.format(model, float)(prompt, max_tokens=10)
    assert isinstance(sequence, float)

    assert sequence != ""
    float(sequence)


def test_llamacpp_integration_bool(model):
    model.model.reset()
    prompt = (
        "<|im_start|>user\nIs this True or False?<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.format(model, bool)(prompt, max_tokens=10)
    assert isinstance(sequence, bool)

    assert sequence != ""
    bool(sequence)


def test_llamacpp_integration_date(model):
    model.model.reset()
    prompt = (
        "<|im_start|>user\nWhat day is it today?<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.format(model, datetime.date)(prompt, max_tokens=10)
    assert isinstance(sequence, datetime.date)


def test_llamacpp_integration_time(model):
    model.model.reset()
    prompt = "<|im_start|>user\nWhat time is it?<|im_end|>\n<|im_start|>assistant\n"
    sequence = generate.format(model, datetime.time)(prompt, max_tokens=10)
    assert isinstance(sequence, datetime.time)


def test_llamacpp_integration_datetime(model):
    model.model.reset()
    prompt = "<|im_start|>user\nWhat time is it?<|im_end|>\n<|im_start|>assistant\n"
    sequence = generate.format(model, datetime.datetime)(prompt, max_tokens=20)
    assert isinstance(sequence, datetime.datetime)


def test_llamacpp_integration_choice(model):
    model.model.reset()
    prompt = (
        "<|im_start|>user\nWrite a short sentence<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.choice(model, ["test", "choice"])(prompt)
    assert sequence == "test" or sequence == "choice"


def test_llamacpp_json_basic(model):
    model.model.reset()
    prompt = "<|im_start|>user\nOutput some JSON<|im_end|>\n<|im_start|>assistant\n"

    class Spam(BaseModel):
        foo: int
        bar: float
        spam: constr(max_length=10)
        fuzz: bool

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)
    result = generate.json(model, Spam)(
        prompt, max_tokens=1000, temperature=0.0, rng=rng
    )
    assert isinstance(result, BaseModel)
    assert isinstance(result.foo, int)
    assert isinstance(result.bar, float)
    assert isinstance(result.spam, str)
    assert isinstance(result.fuzz, bool)
    assert len(result.spam) <= 10


def test_llamacpp_json_schema(model):
    model.model.reset()
    prompt = "<|im_start|>user\nOutput some JSON<|im_end|>\n<|im_start|>assistant\n"

    schema = """{
      "title": "spam",
      "type": "object",
      "properties": {
           "foo" : {"type": "integer"},
           "bar": {"type": "string", "maxLength": 4}
        },
      "required": ["foo", "bar"]
      }
    """

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)
    result = generate.json(model, schema)(
        prompt, max_tokens=500, temperature=0, rng=rng
    )
    assert isinstance(result, dict)
    assert isinstance(result["foo"], int)
    assert isinstance(result["bar"], str)


def test_llamacpp_json_batch(model):
    model.model.reset()
    prompts = [
        "<|im_start|>user\nOutput a valid JSON object. Only use alpha numeric characters as keys.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nOutput a valid JSON object. Only use alpha numeric characters as keys.<|im_end|>\n<|im_start|>assistant\n",
    ]

    class Spam(BaseModel):
        foo: int
        bar: float
        spam: constr(max_length=10)
        fuzz: bool

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)
    result = generate.json(model, Spam)(
        prompts, max_tokens=500, temperature=0.0, rng=rng
    )
    assert isinstance(result[0], BaseModel)
    assert isinstance(result[1], BaseModel)


def test_llamacpp_json_str_enum(model):
    model.model.reset()
    prompt = "<|im_start|>user\nOutput a valid JSON object. Only use alpha numeric characters as keys.<|im_end|>\n<|im_start|>assistant\n"

    class Name(str, Enum):
        john = "John"
        marc = "Marc"
        michel = "Michel"

    class User(BaseModel):
        id: int
        name: Name

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)
    result = generate.json(model, User)(
        prompt, max_tokens=500, temperature=0.0, rng=rng
    )
    assert isinstance(result, BaseModel)
    assert isinstance(result.id, int)
    assert result.name in ["John", "Marc", "Michel"]


def test_llamacpp_json_array(model):
    model.model.reset()
    prompt = "<|im_start|>user\nOutput a valid JSON object. Only use alpha numeric characters as keys.<|im_end|>\n<|im_start|>assistant\n"

    class User(BaseModel):
        id: int
        value: List[float]

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)
    result = generate.json(model, User)(
        prompt,
        max_tokens=500,
        temperature=0.0,
        rng=rng,
        frequency_penalty=0.5,
    )
    assert isinstance(result, BaseModel)
    assert isinstance(result.id, int)
    assert isinstance(result.value, list)
    for value in result.value:
        assert isinstance(value, float) or isinstance(value, int)


def test_llamacpp_json_int_enum(model):
    model.model.reset()
    prompt = "<|im_start|>user\nOutput a valid JSON object. Only use alpha numeric characters as keys.<|im_end|>\n<|im_start|>assistant\n"

    class Id(int, Enum):
        one = 1
        two = 2

    class User(BaseModel):
        id: Id

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)
    result = generate.json(model, User)(
        prompt, max_tokens=500, temperature=0.0, rng=rng
    )
    assert isinstance(result, BaseModel)
    assert isinstance(result.id, int)
    assert result.id in [1, 2]


def test_llamacpp_json_union(model):
    model.model.reset()
    prompt = "<|im_start|>user\nOutput some JSON<|im_end|>\n<|im_start|>assistant\n"

    class Spam(BaseModel):
        foo: int
        bar: Union[constr(max_length=10), float]

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)
    result = generate.json(model, Spam)(
        prompt, max_tokens=100, temperature=0.0, rng=rng
    )
    assert isinstance(result, BaseModel)
    assert (
        isinstance(result.bar, int)
        or isinstance(result.bar, float)
        or isinstance(result.bar, str)
    )


def test_llamacpp_json_function(model):
    model.model.reset()
    prompt = "<|im_start|>user\nOutput arguments for the function, array with 2 elements<|im_end|>\n<|im_start|>assistant\n"

    def function(foo: int, bar: List[int]):
        return foo + sum(bar)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(10)
    sequence = generate.json(model, function)(
        prompt, max_tokens=100, temperature=0.0, rng=rng
    )
    assert isinstance(sequence, dict)
    assert isinstance(function(**sequence), int)


def test_llamacpp_successive_choices(model):
    model.model.reset()

    choose = generate.regex(model, r"(one|two|three)")
    assert choose("pick a numner") in ["one", "two", "three"]

    cities = ["New York", "Paris", "San Francisco"]
    city = generate.choice(model, cities)
    assert city("pick a city") in cities

    assert choose("a number") in ["one", "two", "three"]
