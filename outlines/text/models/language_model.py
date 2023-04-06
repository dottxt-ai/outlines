import inspect

from outlines.graph import Apply, Op
from outlines.text.compose import compose
from outlines.text.var import StringVariable, as_string


class LanguageModel(Op):
    """An `Op` that produces a sample from a language model.

    The output of language models in outlines is represented as a random
    variable.  Therefore, calling a language model will return a random sequence
    (via ancestral sampling) by default. Other decoding methods are constructed
    as graph transformations.

    """

    def __init__(self, name=None):
        """Instantiate the `LanguageModel` `Op`.

        Parameters
        ----------
        name
            The name of the `Op` in the graph.

        """
        super().__init__()
        self.name = name

    def __call__(self, prompt, stops_at=None, name=None):
        """Create the `Apply` node that represents the `Op`'s application to inputs.

        Parameters
        ----------
        prompt
            The prompt used to condition the language model's sampling procedure.
        name
            The name of the output variable in the graph.

        """
        res = super().__call__(prompt)

        if name is not None:
            res.name = name

        return res

    def make_node(self, prompt):
        prompt = as_string(prompt)
        out = StringVariable()

        return Apply(self, [prompt], [out])

    def perform(self, prompt):
        return NotImplementedError


def model(name: str, stops_at=None):
    """Decorator that allows to simplify calls to language models.

    Prompts that are passed to language models are often rendered templates,
    and the workflow typically looks like:

    >>> import outlines
    >>> from outlines.text.models.openai import OpenAI
    >>>
    >>> llm = OpenAI("davinci")
    >>> tpl = "I have a ${question}"
    >>> prompt = outlines.compose(tpl, question="How are you?")
    >>> answer = llm(prompt)

    While explicit, these 4 lines have the following defaults:

    1. The prompt is hidden;
    2. The language model instantiation is far from the prompt; prompt templates
    are however attached to a specific language model call.
    3. The intent behind the language model call is hidden.

    To encapsulate the logic behind language model calls, we thus define the
    template prompt inside a function and decorate the function with a model
    specification. When that function is called, the template is rendered using
    the arguments passed to the function, and the rendered prompt is passed to
    a language model instantiated with the arguments passed to the decorator.

    The previous example is equivalent to the following:

    >>> import outlines
    >>>
    >>> @outlines.text.model("openai/davinci")
    ... def answer(question):
    ...     "I have a ${question}"
    ...
    >>> answer, _ = answer("How are you?")

    Decorated functions return two objects: the first represents the output of
    the language model call, the second represents the concatenation of the
    rendered prompt with the output of the language model call. The latter can
    be used in context where one expands an initial prompt with recursive calls
    to language models.

    """
    provider_name = name.split("/")[0]
    model_name = name[len(provider_name) + 1 :]

    if provider_name == "openai":
        from outlines.text.models.openai import OpenAI

        llm = OpenAI(model_name, stops_at)  # type:ignore
    elif provider_name == "hf":
        from outlines.text.models.hugging_face import HFCausalLM

        llm = HFCausalLM(model_name)  # type:ignore
    else:
        raise NameError(f"The model provider {provider_name} is not available.")

    def decorator(fn):
        # Get the names of the parameters to the function, which must correspond
        # to the variables defined in the template.
        var_names = []
        kwargs_data = {}
        sig = inspect.signature(fn)
        for parameter in sig.parameters.values():
            if parameter.default == inspect._empty:
                var_names.append(parameter.name)
            else:
                kwargs_data[parameter.name] = parameter.default

        # The docstring contains the template that will be rendered to be used
        # as a prompt to the language model.
        template = inspect.cleandoc(fn.__doc__)

        def wrapper(*args, **kwargs):
            """Call the LLM with the rendered template.

            Building prompts with recursive calls to language models is common
            in prompt engineering, we thus return both the raw answer from the
            language model as well as the rendered prompt including the answer.

            Returns
            -------
            A tuple that contains the result of the language model call, and the
            rendered prompt concatenated with the result of the language model
            call.

            """
            args_data = {name: arg for name, arg in zip(var_names, args)}
            kwargs_data.update(kwargs)
            data = {**args_data, **kwargs_data}

            prompt = compose(template, **data)
            result = llm(prompt)
            return result, prompt + result

        return wrapper

    return decorator
