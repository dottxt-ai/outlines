### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import functools
from dataclasses import replace
from typing import Optional

from outlines.models.openai import OpenAI, OpenAIConfig


@functools.singledispatch
def openai(model_or_client, *args, **kwargs):
    if len(args) == 1 and isinstance(args[0], OpenAIConfig):
        return OpenAI(model_or_client, args[0], **kwargs)
    elif kwargs.get("config"):
        return OpenAI(model_or_client, kwargs.pop("config"), **kwargs)
    else:
        return OpenAI(model_or_client, OpenAIConfig(), **kwargs)


@openai.register(str)
def openai_model(
    model_name: str,
    config: Optional[OpenAIConfig] = None,
    **openai_client_params,
):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "The `openai` library needs to be installed in order to use Outlines' OpenAI integration."
        )

    if config is not None:
        config = replace(config, model=model_name)  # type: ignore
    else:
        config = OpenAIConfig(model=model_name)

    client = AsyncOpenAI(**openai_client_params)

    return OpenAI(client, config)


def azure_openai(
    deployment_name: str,
    model_name: Optional[str] = None,
    config: Optional[OpenAIConfig] = None,
    **azure_openai_client_params,
):
    try:
        from openai import AsyncAzureOpenAI
    except ImportError:
        raise ImportError(
            "The `openai` library needs to be installed in order to use Outlines' Azure OpenAI integration."
        )

    if config is not None:
        config = replace(config, model=deployment_name)  # type: ignore
    if config is None:
        config = OpenAIConfig(model=deployment_name)

    client = AsyncAzureOpenAI(**azure_openai_client_params)

    return OpenAI(client, config)
