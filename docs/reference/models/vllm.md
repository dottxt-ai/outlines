# vLLM


!!! Note "Installation"

    You need to install the `vllm` library to use the vLLM integration: `pip install vllm`. The default installation only works on machines with a GPU, follow the [installation section][vllm-install-cpu] for instructions to install vLLM for CPU or ROCm.

    Consult the [vLLM documentation][vllm-docs] for detailed informations about how to initialize OpenAI clients and the available options.

## Load the model

Outlines supports models available via vLLM's offline batched inference interface. You can load a model using:

```python
import outlines
from vllm import LLM

model = outlines.from_vllm(LLM("microsoft/Phi-3-mini-4k-instruct"))
```

Models are loaded from the [HuggingFace hub](https://huggingface.co/).


!!! Warning "Device"

    The default installation of vLLM only allows to load models on GPU. See the [installation instructions][vllm-install-cpu] to run models on CPU.


## Generate text

To generate text, you can just call the model with a prompt as argument:

```python
import outlines
from vllm import LLM

model = outlines.from_vllm(LLM("microsoft/Phi-3-mini-4k-instruct"))
answer = model("Write a short story about a cat.")
```

You can also use structured generation with the `VLLM` model by providing an output type after the prompt:

```python
import outlines
from vllm import LLM
from outlines.types import JsonType
from pydantic import BaseModel

class Character(BaseModel):
    name: str

model = outlines.from_vllm(LLM("microsoft/Phi-3-mini-4k-instruct"))
answer = model("Create a character.", output_type=JsonType(Character))
```

The VLLM model supports batch generation. To use it, you can pass a list of strings as prompt instead of a single string.

## Optional parameters

When calling the model, you can provide optional parameters on top of the prompt and the output type. Those will be passed on to the `LLM.generate` method of the `vllm` library. An optional parameter of particular interest is `sampling_params`, which is an instance of `SamplingParams`. You can find more information about it in the [vLLM documentation][https://docs.vllm.ai/en/latest/api/inference_params.html].

!!! Warning

    Streaming is not available for the offline vLLM integration.

[vllm-docs]:https://docs.vllm.ai/en/latest/
[vllm-install-cpu]: https://docs.vllm.ai/en/latest/getting_started/cpu-installation.html
[vllm-install-rocm]: https://docs.vllm.ai/en/latest/getting_started/amd-installation.html
[rocm-flash-attention]: https://github.com/ROCm/flash-attention/tree/flash_attention_for_rocm#amd-gpurocm-support
