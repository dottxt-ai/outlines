---
title: Welcome to Outlines!
---

# Welcome

![](/assets/images/logo-light-mode.svg#only-light)
![](/assets/images/logo-dark-mode.svg#only-dark)

Outlines is a Python library from [.TXT](https://dottxt.co) that guarantees structured output from
large language models. It ensures LLMs speak the language of your application by making them follow specific formats such as JSON, regular expressions, or context-free grammars.

[Get started](/docs/getting_started){ .md-button .md-button--primary }
[API Reference](/api/reference){ .md-button }
[Examples](/examples){ .md-button }
[GitHub](https://github.com/yourusername/outlines){ .md-button }

## Features

<div class="grid cards" markdown>

- :material-shield-check: **Reliable** - Guaranteed schema compliance -- always valid JSON.
- :material-puzzle: **Feature-rich** - Supports a large proportion of the JSON Schema spec, along with regex and context-free grammars.
- :material-lightning-bolt: **Fast** - Outlines has negligible runtime overhead, and fast compilation times.
- :material-cog: **Universal** - Outlines is a powered by Rust, and can be easily bound to other languages.
- :material-lightbulb: **Simple** - Outlines is a low-abstraction library. Write code the way you normally do with LLMs. No agent frameworks needed.
- :material-magnify: **Powerful** - Manage prompt complexity with prompt templating.

</div>

## Supported models

> [!NOTE]
> Provide full model list with links to docs about each model

- vLLM
- Transformers
- OpenAI

## About .txt

Outlines is built with ❤️ by [.txt](https://dottxt.co).

.txt solves the critical problem of reliable structured output generation for large language models. Our commercially-licensed libraries ensure 100% compliance with JSON Schema, regular expressions and context-free grammars while adding only microseconds of latency. Unlike open-source alternatives, we offer superior reliability, performance, and enterprise support.

## Who is using Outlines?

Hundreds of organisations and the main LLM serving frameworks ([vLLM][vllm], [TGI][tgi], [LoRAX][lorax], [xinference][xinference], [SGLang][sglang]) use Outlines.

Prominent companies and organizations that use Outlines include:

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

</body>

Organizations are included either because they use Outlines as a dependency in a public repository, or because of direct communication between members of the Outlines team and employees at these organizations.

Still not convinced, read [what people say about us](community/feedback.md). And make sure to take a look at what the [community is building](community/examples.md)!

## Philosophy

**Outlines** is a library for neural text generation. You can think of it as a
more flexible replacement for the `generate` method in the
[transformers](https://github.com/huggingface/transformers) library.

**Outlines** helps developers _structure text generation_ to build robust
interfaces with external systems. It provides generation methods that
guarantee that the output will match a regular expressions, or follow
a JSON schema.

**Outlines** provides _robust prompting primitives_ that separate the prompting
from the execution logic and lead to simple implementations of few-shot
generations, ReAct, meta-prompting, agents, etc.

**Outlines** is designed as a _library_ that is meant to be compatible the
broader ecosystem, not to replace it. We use as few abstractions as possible,
and generation can be interleaved with control flow, conditionals, custom Python
functions and calls to other libraries.

**Outlines** is _compatible with every auto-regressive model_. It only interfaces with models
via the next-token logits distribution.

## Outlines people

Outlines would not be what it is today without a community of dedicated developers:

<a href="https://github.com/dottxt-ai/outlines/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dottxt-ai/outlines" />
</a>

## Acknowledgements

<div class="grid" markdown>

<figure markdown>
  <a href="http://www.dottxt.co">
  ![Normal Computing logo](assets/images/dottxt.png){ width="150" }
  </a>
</figure>

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
