# Release Note

### Why a new major version?

The v1 intends on making Outlines more closely focused on constrained generation. To do so, we delegate a wider range of tasks to the users and inference libraries. On top of making Outlines leaner, this design provides more flexibility to the users and let them use interfaces they are already familiar with.

Our approach is inspired by the unix best practices — each element does one thing well, and we compose those functional elements.

As this new version deprecates some previously available features of Outlines, we have written a migration guide that gives detailed information on how to upgrade your v0 code to v1.

### Deprecated

All deprecated features listed below will be removed in version 1.1.0. Until then, a warning will be displayed with information on how to migrate your code to v1.

- The model loader functions from the `models` module (`transformers`, `openai`, etc.) have been deprecated. They are replaced by equivalent functions prefixed with `from_` such as `from_transformers`, `from_openai`, etc. The new loader functions accept different arguments compared to the old ones. They now typically require an instance of an engine/client from the associated inference library. This change was made to avoid duplicating inference library logic and to give users more control over inference engine/client initialization.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/models)

```python
# v0
from outlines import models
from transformers import BertForSequenceClassification, BertTokenizer

model = models.transformers(
    model_name="prajjwal1/bert-tiny",
    model_class=BertForSequenceClassification,
    tokenizer_class=BertTokenizer,
    model_kwargs={"use_cache": False},
    tokenizer_kwargs={"model_max_length": 512},
)

# v1
import outlines
from transformers import BertForSequenceClassification, BertTokenizer

hf_model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", use_cache=False)
hf_tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny", model_max_length=512)
model = outlines.from_transformers(hf_model, hf_tokenizer)
```

- The `generate` module and the associated functions (`json`, `choice`…) have been deprecated. They are replaced by the `Generator` constructor. While you had to select the right generate function for your output type, you can now provide any output type supported by Outlines to the unique `Generator` object.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/core/generator)


```python
# v0
from pydantic import BaseModel
from outlines import generate, models

class Character(BaseModel):
	name: str

model = models.openai("gpt-4o")
generator = generate.json(model, Character)

# v1
from openai import OpenAI
from pydantic import BaseModel
from outlines import Generator, from_openai

class Character(BaseModel):
	name: str

model = from_openai(OpenAI())
generator = Generator(model, Character)
```

- The `TransformersVision` model has been deprecated. It's replaced by `TransformersMultiModal`, which is more general as it supports additional input types beyond images, such as audio. When calling it, instead of providing the prompt and image assets separately, both should now be included in a single dictionary. The model is loaded with `from_transformers` just like the `Transformers` model, but the second argument must be a processor instead of a tokenizer.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/models/transformers_multimodal)


```python
# v0
from io import BytesIO
from urllib.request import urlopen
from PIL import Image
from transformers import LlavaForConditionalGeneration
from outlines import models, generate

def img_from_url(url):
    img_byte_stream = BytesIO(urlopen(url).read())
    return Image.open(img_byte_stream).convert("RGB")

model = models.transformers_vision(
    model_name="trl-internal-testing/tiny-LlavaForConditionalGeneration",
    model_class=LlavaForConditionalGeneration,
)
generator = generate.text(model)
result = generator(
    "Describe the image <image>",
    img_from_url("https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")
)

# v1
from io import BytesIO
from urllib.request import urlopen
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
import outlines

def img_from_url(url):
    img_byte_stream = BytesIO(urlopen(url).read())
    return Image.open(img_byte_stream).convert("RGB")

model = outlines.from_transformers(
	LlavaForConditionalGeneration.from_pretrained("trl-internal-testing/tiny-LlavaForConditionalGeneration"),
	AutoProcessor.from_pretrained("trl-internal-testing/tiny-LlavaForConditionalGeneration")
)
image = img_from_url("https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")
result = model({"text": "Describe the image <image>", "images": image})
```

- The `Exllamav2` model has been deprecated without replacement because its interface is not fully compatible with Outlines. We had to implement cumbersome patching to make it work, so we decided to remove it entirely.

- The `function` module and the associated `Function` class have been deprecated. They are replaced by the `Application` class, which serves a similar purpose to `Function`. There are two notable differences: an `Application` is not initialized with a model (a model must be provided when calling the object), and template variables must be provided in a dictionary instead of as keyword arguments when calling the `Application`.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/utility/application)


```python
# v0
from pydantic import BaseModel
from outlines import Function, Template

class Character(BaseModel):
	name: str

template = Template.from_string("Create a {{ gender }} character.")
fn = Function(template, Character, "hf-internal-testing/tiny-random-GPTJForCausalLM")
response = fn(gender="female")

# v1
from pydantic import BaseModel
from outlines import Application, Template, from_transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

class Character(BaseModel):
	name: str

model = from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

template = Template.from_string("Create a {{ gender }} character.")
app = Application(template, Character)
response = app(model, {"gender": "female"})
```

- The `samplers` module and the associated objects (`multinomial`, `greedy`…) have been deprecated. You should now use the inference arguments specific to the inference library your  model is based on to control the sampling.

```python
# v0
from outlines import generate, models, samplers

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.text(model, samplers.beam_search(2))
response = generator("Write a short story about a cat", max_tokens=10)

# v1
from outlines import Generator, from_transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)
response = model("Write a short story about a cat", num_beams=2)
```

- The `load_lora` methods on the `VLLM` and `LlamaCpp` models have been deprecated. You should now load through the `Llama` instance provided when initializing the model in the case of the `LlamaCpp` model, and provide it as a keyword argument when calling the model in the case of the `VLLM` model.

```python
# v0
from outlines import models
from vllm import LLM

model = models.vllm("erwanf/gpt2-mini")
model.load_lora("path/to/lora/file")
response = model("Write a short story about a cat.")

#v1
from outlines import from_vllm
from vllm import LLM
from vllm.lora.request import LoRARequest

model = from_vllm(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)
lora_request = LoRARequest("path/to/lora/file", 1, "path/to/lora/file")
response = model("Write a short story about a cat.", lora_request=lora_request)
```

### Modified

Some objects are maintained, but their interface or behavior has been modified.

- The interface of `Model` classes (`Transformers`, `OpenAI`, etc.) has been significantly modified. Models can now be called directly with a prompt and an output type without having to create a generator first. Additionally, all models have a `stream` method that can be invoked directly by the user.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/models)


```python
# v0
from pydantic import BaseModel
from outlines import generate, models

class Character(BaseModel):
		name: str

model = models.openai("gpt-4o")
generator = generate.json(model, Character)
result = generator("Create a character")

# v1
from openai import OpenAI
from pydantic import BaseModel
from outlines import from_openai

class Character(BaseModel):
	name: str

model = from_openai(OpenAI(), "gpt-4o")
result = model("Create a character", Character)
```

- The interface of the `__init__` method of the `OpenAI` model class has been modified. While it previously accepted a client and an `OpenAIConfig` object instance, it now accepts a client and a model name. The inference arguments from the config object should now be specified when calling the model to more closely align with the OpenAI Python library's functionality. If you provide an `OpenAIConfig` instance when initializing the model, a deprecation warning will appear and your model will behave like a v0 model.
We recommend using the `from_openai` function instead of initializing models directly.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/models/openai)


```python
# v0
from outlines.models.openai import OpenAI, OpenAIConfig
from openai import OpenAI as OpenAIClient

model = OpenAI(
	OpenAIClient(),
	OpenAIConfig(model="gpt-4o", stop=["."])
)

# v1
import outlines
from openai import OpenAI

model = outlines.from_openai(OpenAIClient(), "gpt-4o")
```

- The return type of text generation is now consistently a string (or list/lists of strings for multiple samples or batching). In v0, Outlines automatically cast the inference result into the type provided by the user for constrained generation, but we have removed this behavior. This change was made to create more consistent behavior and to give users more freedom in deciding how to handle the generation result.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/models)

```python
# v0
from pydantic import BaseModel
from outlines import generate, models

class Character(BaseModel):
	name: str

model = models.openai("gpt-4o")
generator = generate.json(model, Character)
result = generator("Create a character")
print(result) # name='James'

# v1
import openai
from pydantic import BaseModel
from outlines import from_openai

class Character(BaseModel):
		name: str

model = from_openai(OpenAI())
result = model("Create a character", Character)
print(result) # {"name": "James"}
print(Character.model_validate_json(result)) # name='James'
```

- While Outlines was trying to standardize inference argument names across models in v0, we decided to stop doing so and to directly pass on the inference arguments provided by the user to the inference engine/client. Our objective is to let the user use all arguments they are accustomed to with their inference library instead of having to learn Outlines-defined arguments. The deprecation of the `samplers` mentioned above is a part of this change of approach.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/models)

```python
# v0
from outlines import generate, models

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.text(model)
result = generator("Create a character", max_tokens=256, stop_at=".")

# v1
from outlines import from_transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = from_transformers(
	AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
	AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)
result = model("Create a character", max_new_tokens=256, stop_strings=".")
```

### Added features

- There are 8 additional models available. All of them are loaded with an associated `from_` function that accepts an inference engine/client instance.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/models)
    - `Dottxt`
    - `Anthopic`
    - `Gemini`
    - `Ollama`
    - `SGLang`
    - `TGI`
    - `TransformersMultiModel`
    - `VLLM`
- Some server-based models now have an async version. To create an async model, just provide an async client instance when using the loader function. The async models are the following.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/models)
    - `AsyncSGLang`
    - `AsyncTGI`
    - `AsyncVLLM`

```python
import outlines
from huggingface_hub import AsyncInferenceClient

async_model = outlines.from_tgi(AsyncInferenceClient("http://localhost:11434"))
```

- As explained previously, the `Generator` constructor has been added. It accepts a model and an output type as arguments and returns a generator object that can be used to generate text by providing a prompt and inference arguments. The interest of a generator is that it's reusable such that the user does not have to specify the output type they want each time and the output type compilation (when applicable) happens only once.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/core/generator)

```python
# direct model calling
from typing import Literal
from outlines import from_transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = from_transformers(
		AutoModelForCausalLM.from_pretrained("fmicrosoft/Phi-3-mini-4k-instruct"),
		AutoTokenizer.from_pretrained("fmicrosoft/Phi-3-mini-4k-instruct")
)
result = model("Pizza or burger", Literal["pizza", "burger"])

# using a generator
from outlines import Generator, from_transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = from_transformers(
		AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
		AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)
generator = Generator(model, Literal["pizza", "burger"])
result = generator("Pizza or burger")
```

- As explained previously, the `Application` class has been added. An `Application` is initialized with a prompt template and an output type. The application object returned can then be called with a model, a dictionary containing values for the template variables and inference arguments. The objective of this object is to let users easily switch from a model to another for a given set of prompt and output type.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/utility/application)

```python
from pydantic import BaseModel
from outlines import Application, Template

class Character(BaseModel):
	name: str

template = Template.from_string("Create a {{ gender }} character.")
app = Application(template, Character)
response = app(model, {"gender": "female"})
```

- The regex DSL and the associated `Term` classes and functions have been added. Terms (`Regex`, `String`…) can be used as output types to generate text with models or generators (they are turned into a regex). The term functions (`either`, `optional`, `at_least`…) are useful to build more complex regex patterns by combining terms. On top of the objects related to regex patterns, there are also 2 terms that are intended to be used by themselves as output types: `JsonSchema` and `CFG`.
[Documentation](https://dottxt-ai.github.io/outlines/latest/features/core/ouput_types)

```python
# term used directly as an output type
from outlines import from_transformers
from outlines.types import JsonSchema
from transformers import AutoModelForCausalLM, AutoTokenizer

model = from_transformers(
		AutoModelForCausalLM.from_pretrained("fmicrosoft/Phi-3-mini-4k-instruct"),
		AutoTokenizer.from_pretrained("fmicrosoft/Phi-3-mini-4k-instruct")
)
json_schema = '{"type": "object", "properties": {"answer": {"type": "number"}}}'
result = model("What's 2 + 2? Respond in a json", JsonSchema(json_schema))

# creating a complex regex pattern
from outlines import from_transformers
from outlines.types import at_least, either, integer, optional
from transformers import AutoModelForCausalLM, AutoTokenizer

model = from_transformers(
	AutoModelForCausalLM.from_pretrained("fmicrosoft/Phi-3-mini-4k-instruct"),
	AutoTokenizer.from_pretrained("fmicrosoft/Phi-3-mini-4k-instruct")
)
regex_term = "I have " + integer + either("dog", "cat") + optional("s")
result = model("How many pets do you have", regex_term)
```
