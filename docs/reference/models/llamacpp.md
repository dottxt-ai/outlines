# Llama.cpp

Outlines provides an integration with [Llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python library][llamacpp]. Llamacpp allows to run quantized models on machines with limited compute.

!!! Note "Installation"

    You need to install the `llama-cpp-python` library to use the llama.cpp integration. See the [installation section](#installation) for instructions to install `llama-cpp-python` with CUDA, Metal, ROCm and other backends. To get started quickly you can also run:

    ```bash
    pip install "outlines[llamacpp]"
    ```

## Load the model

To load a model you can use the same interface as you would using `llamap-cpp-python` directly. The default method is to initialize the model by passing the path to the weights on your machine. Assuming [Phi2's weights](https://huggingface.co/TheBloke/phi-2-GGUF) are in the current directory:

```python
from outlines import models

llm = models.LlamaCpp("./phi-2.Q4_K_M.gguf")
```

You can initialize the model by passing the name of the repository on the HuggingFace Hub, and the filenames (or glob pattern):


```python
from outlines import models

model = models.LlamaCpp.from_pretrained("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf")
```

This will download the model files to the hub cache folder and load the weights in memory.


You can pass the same keyword arguments to the model as you would pass in the [llama-ccp-library][llamacpp]:

```python
from outlines import models

model = models.LlamaCpp(
    "TheBloke/phi-2-GGUF",
    "phi-2.Q4_K_M.gguf"
    n_ctx=512,  # to set the context length value
)
```

See the [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__) for the full list of parameters.

### Load the model on GPU

!!! Note

    [Make sure](#cuda) that you installed `llama-cpp-python` with GPU support.

 To load the model on GPU, pass `n_gpu_layers=-1`:

```python
from outlines import models

model = models.LlamaCpp(
    "TheBloke/phi-2-GGUF",
    "phi-2.Q4_K_M.gguf",
    n_gpu_layers=-1,  # to use GPU acceleration
)
```


## Generate text


To generate text you must first create a `Generator` object by passing the model instance and, possibley, the expected output type:

```python
from outlines import models, generate


model = models.LlamaCpp("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf")
generator = Generator(model)
```

You can pass to the generator the same keyword arguments you would pass in `llama-cpp-python`:

```python
answer = generator("A prompt", presence_penalty=0.8)
```

You can also stream the tokens:

```python
tokens = generator.stream("A prompt")
```


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
