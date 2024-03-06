# Llama.cpp

!!! Installation

    You need to install the `llama-cpp-python` library to be able to use these models in Outlines.

Outlines provides an integration with [Llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python library][llamacpp]. Llamacpp allows to run quantized models on machines with limited compute.

You can initialize the model by pasing the path to the weights on your machine. Assuming [Phi2's weights](https://huggingface.co/TheBloke/phi-2-GGUF) are in the current directory:

```python
from outlines import models

model = models.llamacpp("./phi-2.Q4_K_M.gguf", device="cuda")
```

If you need more control, you can pass the same keyword arguments to the model as you would pass in the [llama-ccp-library][llamacpp]:

```python
from outlines import models

model = models.llamacpp(
    "./phi-2.Q4_K_M.gguf",
    n_gpu_layers=-1,  # to use GPU acceleration
    seed=1337,  # to set a specific seed
)
```

Please see the [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/) for a list of available keyword arguments. Finally, if for some reason you would like to initialize `llama_cpp.Llama` separately, you can convert it to an Outlines model using:

```python
from llama_cpp import Llama
from outlines import models

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
    filename="*q8_0.gguf",
    verbose=False
)
model = models.LlamaCpp(llm)
```


[llamacpp]: https://github.com/abetlen/llama-cpp-python
