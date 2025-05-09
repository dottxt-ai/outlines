import importlib.util
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import requests

from outlines.v0_legacy import generate
from outlines.v0_legacy.generate.api import GeneratorV0Adapter
from outlines.v0_legacy.models import transformers

if TYPE_CHECKING: # pragma: no cover
    from outlines.templates import Template


def function_warning():
    warnings.warn("""
        The `Function` class is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Instead, you should use the `outlines.Application` class that
        implements a similar functionality.
        An `Application` is initialized with a prompt template and an
        output type. It can then be called with a model and a prompt
        to generate a response that follows the format specified by the
        output type.
        For example:
        ```python
        from pydantic import BaseModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from outlines import models, Application
        from outlines.types import JsonType
        from outlines.templates import Template

        class OutputModel(BaseModel):
            result: int

        model = models.from_transformers(
            AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
            AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        )

        template_string = "What is 2 times {{ num }}?"
        template = Template.from_string(template_string)

        application = Application(template, JsonType(OutputModel))

        result = application(model, num=3)
        print(result)  # Expected output: { "result" : 6 }
        ```
        """,
        DeprecationWarning,
        stacklevel=2
    )


@dataclass
class Function:
    """Represents an Outlines function.

    Functions are a convenient way to encapsulate a prompt template, a language
    model and a Pydantic model that define the output structure. Once defined,
    the function can be called with arguments that will be used to render the
    prompt template.

    """
    prompt_template: "Template"
    schema: Union[str, Callable, object]
    model_name: str
    generator: Optional["GeneratorV0Adapter"] = None

    def __post_init__(self):
        function_warning()

    @classmethod
    def from_github(cls, program_path: str, function_name: str = "fn"):
        """Load a function stored on GitHub"""
        function_warning()
        program_content = download_from_github(program_path)
        function = extract_function_from_file(program_content, function_name)

        return function

    def init_generator(self):
        """Load the model and initialize the generator."""
        model = transformers(self.model_name)
        self.generator = generate.json(model, self.schema)

    def __call__(self, *args, **kwargs):
        """Call the function.

        .. warning::

           This currently does not support batching.

        Parameters
        ----------
        args
            Values to pass to the prompt template as positional arguments.
        kwargs
            Values to pass to the prompt template as keyword arguments.

        """
        # We do not want to raise the deprecation warning about the `json`
        # function or the transformers function here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning
            )
            if self.generator is None:
                self.init_generator()

            prompt = self.prompt_template(*args, **kwargs)
            return self.generator(prompt)


def download_from_github(short_path: str):
    """Download the file in which the function is stored on GitHub."""
    GITHUB_BASE_URL = "https://raw.githubusercontent.com"
    BRANCH = "main"

    path = short_path.split("/")
    if len(path) < 3:
        raise ValueError(
            "Please provide a valid path in the form {USERNAME}/{REPO_NAME}/{PATH_TO_FILE}."
        )
    elif short_path[-3:] == ".py":
        raise ValueError("Do not append the `.py` extension to the program name.")

    username = path[0]
    repo = path[1]
    path_to_file = path[2:]

    url = "/".join([GITHUB_BASE_URL, username, repo, BRANCH] + path_to_file) + ".py"
    result = requests.get(url)

    if result.status_code == 200:
        return result.text
    elif result.status_code == 404:
        raise ValueError(
            f"Program could not be found at {url}. Please make sure you entered the GitHub username, repository name and path to the program correctly."
        )
    else:
        result.raise_for_status()


def extract_function_from_file(content: str, function_name: str) -> Tuple[Callable]:
    """Extract a function object from a downloaded file."""
    spec = importlib.util.spec_from_loader(
        "outlines_function", loader=None, origin="github"
    )
    if spec is not None:  # pragma: no cover
        module = importlib.util.module_from_spec(spec)
        exec(content, module.__dict__)

        try:
            fn = getattr(module, function_name)
        except AttributeError:
            raise AttributeError(
                "Could not find an `outlines.Function` instance in the remote file. Make sure that the path you specified is correct."
            )

        if not isinstance(fn, module.outlines.Function):
            raise TypeError(
                f"The `{function_name}` variable in the program must be an instance of `outlines.Function`"
            )

    return fn
