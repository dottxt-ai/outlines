# Llama.cpp

Outlines provides an integration with [Llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python library][llamacpp]. Llamacpp allows to run quantized models on machines with limited compute.

!!! Note "Documentation"

    To be able to use llama.cpp in Outlines, you must install the `llama-cpp-python` library, `pip install llama-cpp-python`

    Consult the [`llama-cpp-python` documentation](https://llama-cpp-python.readthedocs.io/en/latest) for detailed informations about how to initialize model and the available options.


## Load the model

You can use `outlines.from_llamacpp` to load a `llama-cpp-python` model. Assuming [Phi2's weights](https://huggingface.co/TheBloke/phi-2-GGUF) are in the current directory:

```python
from llama_cpp import Llama
import outlines

llm = outlines.from_llamacpp(Llama("./phi-2.Q4_K_M.gguf"))
```

You can initialize the model by passing the name of the repository on the HuggingFace Hub, and the filenames (or glob pattern):


```python
from llama_cpp import Llama
import outlines

model = outlines.from_llamacpp(Llama.from_pretrained("TheBloke/phi-2-GGUF", "phi-2.Q4_K_M.gguf"))
```

This will download the model files to the hub cache folder and load the weights in memory. See the [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__) for the full list of parameters you can pass to the `Llama` class.


### Load the model on GPU

!!! Note

    [Make sure](#cuda) that you installed `llama-cpp-python` with GPU support.

 To load the model on GPU, pass `n_gpu_layers=-1`:

```python
from llama_cpp import Llama
import outlines

model = outlines.from_llamacpp(Llama(
    "TheBloke/phi-2-GGUF",
    "phi-2.Q4_K_M.gguf",
    n_gpu_layers=-1,  # to use GPU acceleration
))
```


## Generate text


To generate text you must first create a `Generator` object by passing the model instance and, possibley, the expected output type:

```python
import outlines


model = outlines.from_llamacpp(Llama(
    "TheBloke/phi-2-GGUF",
    "phi-2.Q4_K_M.gguf",
    n_gpu_layers=-1,  # to use GPU acceleration
))
generator = outlines.Generator(model)
```

You can pass to the generator the same keyword arguments you would pass in `llama-cpp-python`:

```python
answer = generator("A prompt", presence_penalty=0.8)
```

You can also stream the tokens:

```python
tokens = generator.stream("A prompt")
```


[llamacpp]: https://github.com/abetlen/llama-cpp-python
[llama-cpp-python-call]: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__
[llama-cpp-python-install]: https://github.com/abetlen/llama-cpp-python/tree/08b16afe11e7b42adec2fed0a781123383476045?tab=readme-ov-file#supported-backends
