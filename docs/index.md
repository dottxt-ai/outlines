---
title: Welcome to Outlines!
hide:
  - navigation
---

#

<figure markdown>
![](assets/images/logo-light-mode.svg#only-light){ width="500" }
![](assets/images/logo-dark-mode.svg#only-dark){ width="500" }
</figure>

LLMs are powerful but their outputs are unpredictable. Most solutions attempt to fix bad outputs after generation using parsing, regex, or fragile code that breaks easily.

Outlines guarantees structured outputs during generation — directly from any LLM.

- **Works with any model** - Same code runs across OpenAI, Ollama, vLLM, and more
- **Simple integration** - Just pass your desired output type: `model(prompt, output_type)`
- **Guaranteed valid structure** - No more parsing headaches or broken JSON
- **Provider independence** - Switch models without changing code
- **Rich structure definition** - Use Json Schema, regular expressions or context-free grammars

<figure markdown>
[Get Started](guide/getting_started){ .md-button .md-button--primary }
[View Examples](examples/){ .md-button }
[API Reference](api_reference/){ .md-button }
[GitHub](https://github.com/dottxt-ai/outlines){ .md-button }
</figure>

## See it in action

```python
from pydantic import BaseModel
from typing import Literal
import outlines
import openai

class Customer(BaseModel):
    name: str
    urgency: Literal["high", "medium", "low"]
    issue: str

client = openai.OpenAI()
model = outlines.from_openai(client, "gpt-4o")

customer = model(
    "Alice needs help with login issues ASAP",
    Customer
)
# ✓ Always returns valid Customer object
# ✓ No parsing, no errors, no retries
```

## Quick install

```shell
pip install outlines
```

## Features

<div class="grid cards" markdown>

- :material-shield-check: **Reliable** - Guaranteed schema compliance -- always valid JSON.
- :material-puzzle: **Feature-rich** - Supports a large proportion of the JSON Schema spec, along with regex and context-free grammars.
- :material-lightning-bolt: **Fast** - Microseconds of overhead vs seconds of retries. Compilation happens once, not every request.
- :material-lightbulb: **Simple** - Outlines is a low-abstraction library. Write code the way you normally do with LLMs. No agent frameworks needed.

</div>

## Supported inference APIs, libraries & servers

- [vLLM](features/models/vllm.md)
- [vLLM offline](features/models/vllm_offline.md)
- [Transformers](features/models/transformers.md)
- [llama.cpp](features/models/llamacpp.md)
- [Ollama](features/models/ollama.md)
- [MLX-LM](features/models/mlxlm.md)
- [SgLang](features/models/sglang.md)
- [TGI](features/models/tgi.md)
- [OpenAI](features/models/openai.md)
- [Anthropic](features/models/anthropic.md)
- [Gemini](features/models/gemini.md)
- [Dottxt](features/models/dottxt.md)

## Who is using Outlines?

Hundreds of organisations and the main LLM serving frameworks ([vLLM][vllm], [TGI][tgi], [LoRAX][lorax], [xinference][xinference], [SGLang][sglang]) use Outlines. Prominent companies and organizations that use Outlines include:

<div class="grid cards" markdown>
  <div class="row"><img src="../logos/amazon.png" width="200"></div>
  <div class="row"><img src="../logos/apple.png" width="200"></div>
  <div class="row"><img src="../logos/best_buy.png" width="200"></div>
  <div class="row"><img src="../logos/canoe.png" width="200"></div>
  <div class="row"><img src="../logos/cisco.png" width="200"></div>
  <div class="row"><img src="../logos/dassault_systems.png" width="200"></div>
  <div class="row"><img src="../logos/databricks.png" width="200"></div>
  <div class="row"><img src="../logos/datadog.png" width="200"></div>
  <div class="row"><img src="../logos/dbt_labs.png" width="200"></div>
  <div class="row"><img src="../assets/images/dottxt.png" width="200"></div>
  <div class="row"><img src="../logos/gladia.jpg" width="200"></div>
  <div class="row"><img src="../logos/harvard.png" width="200"></div>
  <div class="row"><img src="../logos/hf.png" width="200"></div>
  <div class="row"><img src="../logos/johns_hopkins.png" width="200"></div>
  <div class="row"><img src="../logos/meta.png" width="200"></div>
  <div class="row"><img src="../logos/mit.png" width="200"></div>
  <div class="row"><img src="../logos/mount_sinai.png" width="200"></div>
  <div class="row"><img src="../logos/nvidia.png" width="200"></div>
  <div class="row"><img src="../logos/nyu.png" width="200"></div>
  <div class="row"><img src="../logos/safran.png" width="200"></div>
  <div class="row"><img src="../logos/salesforce.png" width="200"></div>
  <div class="row"><img src="../logos/shopify.png" width="200"></div>
  <div class="row"><img src="../logos/smithsonian.png" width="200"></div>
  <div class="row"><img src="../logos/tinder.png" width="200"></div>
  <div class="row"><img src="../logos/upenn.png" width="200"></div>
</div>

Organizations are included either because they use Outlines as a dependency in a public repository, or because of direct communication between members of the Outlines team and employees at these organizations.

Still not convinced, read [what people say about us](community/feedback.md). And make sure to take a look at what the [community is building](community/examples.md)!


## Outlines people

Outlines would not be what it is today without a community of dedicated developers:

<a href="https://github.com/dottxt-ai/outlines/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dottxt-ai/outlines" />
</a>

## About .txt

Outlines is built with ❤️ by [.txt](https://dottxt.co).

.txt solves the critical problem of reliable structured output generation for large language models. Our [commercially-licensed libraries][dottxt-doc] ensure 100% compliance with JSON Schema, regular expressions and context-free grammars while adding only microseconds of latency. Unlike open-source alternatives, we offer superior reliability, performance, and enterprise support.


## Acknowledgements

<div class="grid" markdown>

<figure markdown>
  <a href="https://www.normalcomputing.ai">
  ![Normal Computing logo](assets/images/normal_computing.jpg){ width="150" }
  </a>
</figure>

</div>

Outlines was originally developed at [@NormalComputing](https://twitter.com/NormalComputing) by [@remilouf](https://twitter.com/remilouf) and [@BrandonTWillard](https://twitter.com/BrandonTWillard). It is now maintained by [.txt](https://dottxt.co).

[discord]: https://discord.gg/R9DSu34mGd
[aesara]: https://github.com/aesara-devs
[blackjax]: https://github.com/blackjax-devs/blackjax
[pythological]: https://github.com/pythological
[hy]: https://hylang.org/
[.txt]: https://dottxt.co
[vllm]: https://github.com/vllm-project/vllm
[tgi]: https://github.com/huggingface/text-generation-inference
[lorax]: https://github.com/predibase/lorax
[xinference]: https://github.com/xorbitsai/inference
[sglang]: https://github.com/sgl-project/sglang/
[dottxt-doc]: https://docs.dottxt.co
