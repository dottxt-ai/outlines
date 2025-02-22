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

In addition to the parameters described in the [text generation section](../text.md) you can pass an instance of `SamplingParams` directly to any generator via the `sampling_params` keyword argument:

```python
from vllm.sampling_params import SamplingParams
from outlines import models, generate


model = outlines.from_vllm(LLM("microsoft/Phi-3-mini-4k-instruct"))
params = SamplingParams(n=2, frequency_penalty=1., min_tokens=2)
answer = model("A prompt", sampling_params=params)
```

!!! Warning

    Streaming is not available for the offline vLLM integration.

[vllm-docs]:https://docs.vllm.ai/en/latest/
[vllm-install-cpu]: https://docs.vllm.ai/en/latest/getting_started/cpu-installation.html
[vllm-install-rocm]: https://docs.vllm.ai/en/latest/getting_started/amd-installation.html
[rocm-flash-attention]: https://github.com/ROCm/flash-attention/tree/flash_attention_for_rocm#amd-gpurocm-support
