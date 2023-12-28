import pytest
import responses
from pydantic import BaseModel
from requests.exceptions import HTTPError

import outlines
from outlines.function import Function, download_from_github, extract_function_from_file


def test_function_basic():
    @outlines.prompt
    def test_template(text: str):
        """{{ text }}"""

    class Foo(BaseModel):
        id: int

    fn = Function(test_template, Foo, "hf-internal-testing/tiny-random-GPTJForCausalLM")

    assert fn.generator is None

    result = fn("test")
    assert isinstance(result, BaseModel)


def test_download_from_github_invalid():
    with pytest.raises(ValueError, match="Please provide"):
        download_from_github("outlines/program")

    with pytest.raises(ValueError, match="Do not append"):
        download_from_github("outlines-dev/outlines/program.py")


@responses.activate
def test_download_from_github_success():
    responses.add(
        responses.GET,
        "https://raw.githubusercontent.com/outlines-dev/outlines/main/program.py",
        body="import outlines\n",
        status=200,
    )

    file = download_from_github("outlines-dev/outlines/program")
    assert file == "import outlines\n"

    responses.add(
        responses.GET,
        "https://raw.githubusercontent.com/outlines-dev/outlines/main/foo/bar/program.py",
        body="import outlines\n",
        status=200,
    )

    file = download_from_github("outlines-dev/outlines/foo/bar/program")
    assert file == "import outlines\n"


@responses.activate
def test_download_from_github_error():
    responses.add(
        responses.GET,
        "https://raw.githubusercontent.com/foo/bar/main/program.py",
        json={"error": "not found"},
        status=404,
    )

    with pytest.raises(ValueError, match="Program could not be found at"):
        download_from_github("foo/bar/program")

    responses.add(
        responses.GET,
        "https://raw.githubusercontent.com/foo/bar/main/program.py",
        json={"error": "Internal Server Error"},
        status=500,
    )

    with pytest.raises(HTTPError, match="500 Server Error"):
        download_from_github("foo/bar/program")


def test_extract_function_from_file():
    content = """
import outlines
from pydantic import BaseModel

model = "gpt2"


@outlines.prompt
def prompt():
    '''Hello'''


class User(BaseModel):
    id: int
    name: str


function = outlines.Function(
    prompt,
    User,
    "gpt2",
)
    """

    fn = extract_function_from_file(content, "function")
    assert (
        str(type(fn)) == "<class 'outlines.function.Function'>"
    )  # because imported via `exec`


def test_extract_function_from_file_no_function():
    content = """
import outlines
from pydantic import BaseModel

@outlines.prompt
def prompt():
    '''Hello'''


class User(BaseModel):
    id: int
    name: str

program = outlines.Function(
    prompt,
    User,
    "gpt2",
)
    """

    with pytest.raises(AttributeError, match="Could not find"):
        extract_function_from_file(content, "function")
