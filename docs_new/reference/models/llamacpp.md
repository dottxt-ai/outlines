# Llama.cpp

Outlines provides an integration with [Llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python library][llamacpp]. Llamacpp allows to run quantized models on machines with limited compute.

!!! Note "Installation"

    You need to install the `llama-cpp-python` library to use the llama.cpp integration. See the [installation section](#installation) for instructions to install `llama-cpp-python` with CUDA, Metal, ROCm and other backends. To get started quickly you can also run:

    ```bash
    pip install "outlines[llamacpp]"
    ```

## Load the model

You can initialize the model by passing the name of the repository on the HuggingFace Hub, and the filenames (or glob pattern):

```python
from outlines import models

model = models.llamacpp("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf")
```

This will download the model files to the hub cache folder and load the weights in memory.

You can also initialize the model by passing the path to the weights on your machine. Assuming [Phi2's weights](https://huggingface.co/TheBloke/phi-2-GGUF) are in the current directory:

```python
from outlines import models
from llama_cpp import Llama

llm = Llama("./phi-2.Q4_K_M.gguf")
model = models.LlamaCpp(llm)
```

If you need more control, you can pass the same keyword arguments to the model as you would pass in the [llama-ccp-library][llamacpp]:

```python
from outlines import models

model = models.llamacpp(
    "TheBloke/phi-2-GGUF",
    "phi-2.Q4_K_M.gguf"
    n_ctx=512,  # to set the context length value
)
```

**Main parameters:**

| Parameters | Type | Description | Default |
|------------|------|-------------|---------|
| `n_gpu_layers`| `int` | Number of layers to offload to GPU. If -1, all layers are offloaded | `0` |
| `split_mode` | `int` | How to split the model across GPUs. `1` for layer-wise split, `2` for row-wise split | `1` |
| `main_gpu` | `int` | Main GPU | `0` |
| `tensor_split` | `Optional[List[float]]` | How split tensors should be distributed across GPUs. If `None` the model is not split. | `None` |
| `n_ctx` | `int` | Text context. Inference from the model if set to `0` | `0` |
| `n_threads` | `Optional[int]` | Number of threads to use for generation. All available threads if set to `None`.| `None` |
| `verbose` | `bool` | Print verbose outputs to `stderr` | `False` |

See the [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__) for the full list of parameters.

### Load the model on GPU

!!! Note

    [Make sure](#cuda) that you installed `llama-cpp-python` with GPU support.

 To load the model on GPU, pass `n_gpu_layers=-1`:

```python
from outlines import models

model = models.llamacpp(
    "TheBloke/phi-2-GGUF",
    "phi-2.Q4_K_M.gguf",
    n_gpu_layers=-1,  # to use GPU acceleration
)
```

This also works with generators built with `generate.regex`, `generate.json`, `generate.cfg`, `generate.format` and `generate.choice`.

### Load LoRA adapters

You can load LoRA adapters dynamically:

```python
from outlines import models, generate

model = models.llamacpp("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf")
generator = generate.text(model)
answer_1 = generator("prompt")

model.load_lora("./path/to/adapter.gguf")
answer_2 = generator("prompt")
```

To load another adapter you need to re-initialize the model. Otherwise the adapter will be added on top of the previous one:

```python
from outlines import models

model = models.llamacpp("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf")
model.load_lora("./path/to/adapter1.gguf")  # Load first adapter

model = models.llamacpp("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf")
model.load_lora("./path/to/adapter2.gguf")  # Load second adapter
```

## Generate text

In addition to the parameters described in the [text generation section](../text.md) you can pass extra keyword arguments, for instance to set sampling parameters not exposed in Outlines' public API:

```python
from outlines import models, generate


model = models.llamacpp("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf")
generator = generate.text(model)

answer = generator("A prompt", presence_penalty=0.8)
```

**Extra keyword arguments:**

The value of the keyword arguments you pass to the generator suspersede the values set when initializing the sampler or generator. All extra sampling methods and repetition penalties are disabled by default.

| Parameters | Type | Description | Default |
|------------|------|-------------|---------|
| `suffix` | `Optional[str]` | A suffix to append to the generated text. If `None` no suffix is added. | `None` |
| `echo` | `bool` | Whether to preprend the prompt to the completion. | `False` |
| `seed` | `int` | The random seed to use for sampling. | `None` |
| `max_tokens` | `Optional[int]` | The maximum number of tokens to generate. If `None` the maximum number of tokens depends on `n_ctx`. | `16` |
| `frequence_penalty` | `float` | The penalty to apply to tokens based on their frequency in the past 64 tokens. | `0.0` |
| `presence_penalty` | `float` | The penalty to apply to tokens based on their presence in the past 64 tokens. | `0.0` |
| `repeat_penalty` | `float` | The penalty to apply to repeated tokens in the past 64 tokens. | `1.` |
| `stopping_criteria` | `Optional[StoppingCriteriaList]` | A list of stopping criteria to use. | `None`
| `logits_processor` | `Optional[LogitsProcessorList]` | A list of logits processors to use. The logits processor used for structured generation will be added to this list. | `None`
| `temperature` | `float` | The temperature to use for sampling | `1.0` |
| `top_p` | `float` | The top-p value to use for [nucleus sampling][degeneration]. | `1.` |
| `min_p` | `float` | The min-p value to use for [minimum-p sampling][minimum-p]. | `0.` |
| `typical_p` | `float` | The p value to use for [locally typical sampling][locally-typical]. | `1.0` |
| `stop` | `Optional[Union[str, List[str]]]` | A list of strings that stop generation when encountered. | `[]` |
| `top_k` |  `int` | The top-k value used for [top-k sampling][top-k]. Negative value to consider all logit values. | `-1.` |
| `tfs_z` | `float` | The [tail-free sampling][tail-free] parameter. | `1.0` |
| `mirostat_mode` | `int` | The [mirostat sampling][mirostat] mode. | `0` |
| `mirostat_tau` | `float` | The target cross-entropy for [mirostat sampling][mirostat].| `5.0` |
| `mirostat_eta` | `float` | The learning rate used to update `mu` in [mirostat sampling][mirostat]. | `0.1` |

See the [llama-cpp-python documentation][llama-cpp-python-call] for the full and up-to-date list of parameters and the [llama.cpp code][llama-cpp-sampling-params] for the default values of other
sampling parameters.

### Streaming


## Installation

You need to install the `llama-cpp-python` library to use the llama.cpp integration.

### CPU

For a *CPU-only* installation run:

```bash
pip install llama-cpp-python
```

!!! Warning

    Do not run this command if you want support for BLAS, Metal or CUDA. Follow the instructions below instead.

### CUDA

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

It is also possible to install pre-built wheels with CUDA support (Python 3.10 and above):

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version>
```

Where `<cuda-version>` is one of the following, depending on the version of CUDA installed on your system:

- `cu121` for CUDA 12.1
- `cu122` for CUDA 12.2
- `cu123` CUDA 12.3

### Metal

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

It is also possible to install pre-build wheels with Metal support (Python 3.10 or above, MacOS 11.0 and above):

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

### OpenBLAS

```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

### Other backend

`llama.cpp` supports many other backends. Refer to the [llama.cpp documentation][llama-cpp-python-install] to use the following backends:

- CLBast (OpenCL)
- hipBLAS (ROCm)
- Vulkan
- Kompute
- SYCL




[llamacpp]: https://github.com/abetlen/llama-cpp-python
[llama-cpp-python-call]: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__
[llama-cpp-python-install]: https://github.com/abetlen/llama-cpp-python/tree/08b16afe11e7b42adec2fed0a781123383476045?tab=readme-ov-file#supported-backends
[llama-cpp-sampling-params]: https://github.com/ggerganov/llama.cpp/blob/e11a8999b5690f810c2c99c14347f0834e68c524/common/sampling.h#L22
[mirostat]: https://arxiv.org/abs/2007.14966
[degeneration]: https://arxiv.org/abs/1904.09751
[top-k]: https://arxiv.org/abs/1805.04833
[minimum-p]: https://github.com/ggerganov/llama.cpp/pull/3841
[locally-typical]: https://arxiv.org/abs/2202.00666
[tail-free]: https://www.trentonbricken.com/Tail-Free-Sampling
