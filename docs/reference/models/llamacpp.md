# Llama.cpp

!!! Installation

    You need to install the `llama-cpp-python` library to be able to use these models in Outlines.

Outlines provides an integration with [Llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python library](https://github.com/abetlen/llama-cpp-python). Llamacpp allows to run quantized models on machines with limited compute.

Assuming [Phi2's weights](https://huggingface.co/TheBloke/phi-2-GGUF) are in the current directory:

```python
from outlines import models, generate

model = models.llamacpp("./phi-2.Q4_K_M.gguf", device="cpu")
```
