import pytest
import responses
from pydantic import BaseModel
from requests.exceptions import HTTPError

import outlines
from outlines.v0_legacy.function import (
    Function,
    download_from_github,
    extract_function_from_file,
)


def test_function_basic():
    with pytest.warns(DeprecationWarning, match="The @prompt decorator"):
        with pytest.warns(
            DeprecationWarning, match="The `Function` class is deprecated"
        ):
            @outlines.prompt
            def test_template(text: str):
                """{{ text }}"""

            class Foo(BaseModel):
                id: int

            fn = Function(
                test_template,
                Foo,
                "erwanf/gpt2-mini",
            )

            assert fn.generator is None

            result = fn("test")
            assert isinstance(result, BaseModel)
            assert fn.generator is not None
            fn("test")


@responses.activate
def test_function_from_github():
    content = """
import outlines
from pydantic import BaseModel

model = "gpt2"


prompt = outlines.Template.from_string("{{ text }}")


class User(BaseModel):
    id: int
    name: str


fn = outlines.Function(
    prompt,
    User,
    "gpt2",
)
"""
    responses.add(
        responses.GET,
        "https://raw.githubusercontent.com/dottxt-ai/outlines/main/program.py",
        body=content,
        status=200,
    )
    with pytest.warns(
        DeprecationWarning,
        match="The `Function` class is deprecated",
    ):
        fn = Function.from_github("dottxt-ai/outlines/program", "fn")
        assert (
            str(type(fn)) == "<class 'outlines.v0_legacy.function.Function'>"
        )


def test_download_from_github_invalid():
    with pytest.raises(ValueError, match="Please provide"):
        download_from_github("outlines/program")

    with pytest.raises(ValueError, match="Do not append"):
        download_from_github("dottxt-ai/outlines/program.py")


@responses.activate
def test_download_from_github_success():
    responses.add(
        responses.GET,
        "https://raw.githubusercontent.com/dottxt-ai/outlines/main/program.py",
        body="import outlines\n",
        status=200,
    )

    file = download_from_github("dottxt-ai/outlines/program")
    assert file == "import outlines\n"

    responses.add(
        responses.GET,
        "https://raw.githubusercontent.com/dottxt-ai/outlines/main/foo/bar/program.py",
        body="import outlines\n",
        status=200,
    )

    file = download_from_github("dottxt-ai/outlines/foo/bar/program")
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
    with pytest.raises(
        DeprecationWarning,
        match="The @prompt decorator is deprecated",
    ):
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
    with pytest.raises(
        DeprecationWarning,
        match="The @prompt decorator is deprecated",
    ):
        with pytest.raises(AttributeError, match="Could not find"):
            extract_function_from_file(content, "function")
