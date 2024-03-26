# vLLM


!!! Note "Installation"

    You need to install the `vllm` library to use the vLLM integration.

## Load the model

Outlines supports models available via vLLM's offline batched inference interface. You can load a model using:


```python
from outlines import models

model = models.vllm("mistralai/Mistral-7B-v0.1")
```

Or alternatively:

```python
import vllm
from outlines import models

llm = vllm.LLM("mistralai/Mistral-7B-v0.1")
model = models.VLLM(llm)
```


Models are loaded from the [HuggingFace hub](https://huggingface.co/).


!!! Warning "Device"

    vLLM models can only be loaded on GPU.


You can pass any parameter that you would normally pass to `vllm.LLM`, as keyword arguments:

```python
from outlines import models

model = models.vllm(
    "mistralai/Mistral-7B-v0.1",
    trust_remote_code=True,
    gpu_memory_utilization=0.7
)
```

**Main parameters:**

| **Parameters** | **Type** | **Description** | **Default** |
|----------------|:---------|:----------------|:------------|
| `tokenizer_mode`| `str`  | "auto" will use the fast tokenizer if available and "slow" will always use the slow tokenizer. | `auto`
| `trust_remote_code`| `bool` | Trust remote code when downloading the model and tokenizer. | `False` |
| `tensor_parallel_size`| `int` | The number of GPUs to use for distributed execution with tensor parallelism.| `1` |
| `dtype`| `str` | The data type for the model weights and activations. Currently, we support `float32`, `float16`, and `bfloat16`. If `auto`, we use the `torch_dtype` attribute specified in the model config file. However, if the `torch_dtype` in the config is `float32`, we will use `float16` instead.| `auto` |
| `quantization`| `Optional[str]` | The method used to quantize the model weights. Currently, we support "awq", "gptq" and "squeezellm". If None, we first check the `quantization_config` attribute in the model config file. If that is None, we assume the model weights are not quantized and use `dtype` to determine the data type of the weights.| `None` |
| `revision`| `Optional[str]` | The specific model version to use. It can be a branch name, a tag name, or a commit id.| `None` |
| `tokenizer_revision`| `Optional[str]`| The specific tokenizer version to use. It can be a branch name, a tag name, or a commit id.| `None` |
| `gpu_memory_utilization`| `float` | The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache. Higher values will increase the KV cache size and thus improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors.| `0.9` |
| `swap_space`| `int` | The size (GiB) of CPU memory per GPU to use as swap space. This can be used for temporarily storing the states of the requests when their `best_of` sampling parameters are larger than 1. If all requests will have `best_of=1`, you can safely set this to 0. Otherwise, too small values may cause out-of-memory (OOM) errors.| 4 |
| `enforce_eager`| `bool` | Whether to enforce eager execution. If True, we will disable CUDA graph and always execute the model in eager mode. If False, we will use CUDA graph and eager execution in hybrid.| `False` |
| `enable_lora` | `bool` | Whether to enable loading LoRA adapters | `False` |

See the [vLLM code](https://github.com/vllm-project/vllm/blob/8f44facdddcf3c704f7d6a2719b6e85efc393449/vllm/entrypoints/llm.py#L72) for a list of all the available parameters.

### Use quantized models

vLLM supports AWQ, GPTQ and SqueezeLLM quantized models:


```python
from outlines import models

model = models.vllm("TheBloke/Llama2-7b-Chat-AWQ", quantization="awq")
model = models.vllm("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", quantization="gptq")
model = models.vllm("https://huggingface.co/squeeze-ai-lab/sq-llama-30b-w4-s5", quantization="squeezellm")
```

!!! Warning "Dependencies"

    To use AWQ model you need to install the auto-awq library `pip install auto-awq`.

    To use GPTQ models you need to install the autoGTPQ and optimum libraries `pip install auto-gptq optimum`.

### Load LoRA adapters

You can load LoRA adapters and alternate between them dynamically:

```python
from outlines import models

model = models.vllm("facebook/opt-350m", enable_lora=True)
model.load_lora("ybelkaa/opt-350m-lora")  # Load LoRA adapter
model.load_lora(None)  # Unload LoRA adapter
```

## Generate text

In addition to the parameters described in the [text generation section](text.md) you can pass an instance of `SamplingParams` directly to any generator via the `sampling_params` keyword argument:

```python
from vllm.sampling_params import SamplingParams
from outlines import models, generate


model = models.vllm("mistralai/Mistral-7b-v0.1")
generator = generate.text(model)

params = SamplingParams(n=2, length_penalty=1., min_tokens=2)
answer = generator("A prompt", sampling_params=params)
```

This also works with `generate.regex`, `generate.json`, `generate.cfg`, `generate.format` and `generate.choice`.

!!! Note

    The values passed via the `SamplingParams` instance supersede the other arguments to the generator or the samplers.

**`SamplingParams` attributes:**

| Parameters | Type             | Description            | Default |
|:-----------|------------------|:-----------------------|---------|
| `n` | `int` | Number of output sequences to return for the given prompt. | `1` |
| `best_of` | `Optional[int]` | Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This is treated as the beam width when `use_beam_search` is True. By default, `best_of` is set to `n`. | `None` |
| `presence_penalty` | `float` | Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.| `0.0` |
| `frequency_penalty` | `float` | Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens. | `0.0`
| `repetition_penalty` | `float` | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens. | `1.0` |
| `temperature` | `float` | Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling. | `1.0` |
| `top_p` | `float` |  Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens. | `1.0` |
| `top_k` | `int` |  Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens. | `-1` |
| `min_p` |`float` |  Float that represents the minimum probability for a token to be considered, relative to the probability of the most likely token. Must be in [0, 1]. Set to 0 to disable this. | `0.0` |
| `seed` | `Optional[int]` | Random seed to use for the generation. | `None` |
| `use_beam_search` | `bool` |  Whether to use beam search instead of sampling. | `False` |
| `length_penalty` | `float` | Float that penalizes sequences based on their length. Used in beam search.  | `1.0` |
| `early_stopping` | `Union[bool, str]` |  Controls the stopping condition for beam search. It accepts the following values: `True`, where the generation stops as soon as there are `best_of` complete candidates; `False`, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm). | `False` |
| `stop` | `Optional[Union[str, List[str]]]` |  List of strings that stop the generation when they are generated. The returned output will not contain the stop strings. | `None` |
| `stop_token_ids` | `Optional[List[int]]` |  List of tokens that stop the generation when they are generated. The returned output will contain the stop tokens unless the stop tokens are special tokens. | `None` |
| `include_stop_str_in_output` | `bool` |  Whether to include the stop strings in output text. Defaults to False. | `False` |
| `ignore_eos` | `bool` |  Whether to ignore the EOS token and continue generating tokens after the EOS token is generated. | `False` |
| `max_tokens` | `int` |  Maximum number of tokens to generate per output sequence. | `16` |
| `min_tokens` | `int` | Minimum number of tokens to generate per output sequence before EOS or stop_token_ids can be generated | `0` |
| `skip_special_tokens` | `bool` | Whether to skip special tokens in the output. | `True` |
| `spaces_between_special_tokens` | `bool` |  Whether to add spaces between special tokens in the output.  Defaults to True. | `True` |
