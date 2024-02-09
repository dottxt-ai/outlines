---
date: 2024-01-10
categories:
  - Roadmap
authors:
  - remilouf
---

# Roadmap for 2024

Outlines is not even one year old and it's already gone a long way! As we just reached 4000 stars, and before laying out the roadmap for the following year, we would like to pause and thank all of you for supporting us, using and contributing to the library!

![4000 stars](../assets/4000_stars.png)

## Thoughts

Before delving into [the detailed roadmap](#detailed-roadmap), let me share a few thoughts and explain the general direction of the library. These thoughts are informed with my multiple interactions with users, either on [Twitter](https://twitter.com/remilouf) or in our [Discord server](https://discord.gg/ZxBxyWmW5n).

*Outlines currently differentiates itself* from other libraries with its efficient JSON- and regex- constrained generation. A user-facing interface for grammar-structured generation (it had been hidden in the repository) was also recently added. But there is much more we can do along these lines. In 2024 will we will keep pushing in the direction of more accurate, faster constrained generation.

Outlines also supports many models providers: `transformers`, `mamba`, `llama.cpp` and `exllama2`. Those *integrations represent a lot of maintenance*, and we will need to simplify them. For instance, `transformers` now supports quantized models, and we will soon deprecate the support for `autoawq` and `autogptq`.
Thanks to a refactor of the library, it is now possible to use our constrained generation method by using logits processor with all other libraries, except `mamba`. We will look for libraries that provide state-space models and allow to pass a logits processor during inference. We will interface with `llama.cpp` and `exllama2` using logits processors.

*We would like expand our work to the whole sampling layer*, and add new sampling methods that should make structured generation more accurate. This means we will keep the `transformers` integration as it is today and will expand our text generation logic around this library.

Making workflows re-usable and easy to share is difficult today. That is why *we are big believers in [outlines functions](https://github.com/outlines-dev/functions)*. We will keep improving the interface and adding examples.

Finally, *we want to add a CLI tool*, `outlines serve`. This will allows you to either serve an API that does general constrained generation, or to serve Outlines function.

## Detailed roadmap

Here is a more detailed roadmap for the next 12 months. Outlines is a [community](https://discord.gg/ZxBxyWmW5n) effort, and we invite you to pick either topic and [contribute to the library](https://github.com/outlines-dev/outlines). I will progressively add related [issues](https://github.com/outlines-dev/outlines/issues) in the repository.

### Many more examples and tutorials

Let's be honest, Outlines is lacking clear and thorough examples. We want to change this!

* How does Outlines work? What can you do with it?
* What can you do with Outlines that is harder or impossible to do with other libraries?
* How you can perform standard LLM workflows, for instance Chain of Thoughts, Tree of Thoughts, etc?
* How does Oultines integrates with the larger ecosystem, for instance other libraries like LangChain and LlamaIndex?

### Simplify the integrations

We want to keep the current integrations but lower the maintenance cost so we can focus on what we bring to the table.

* Deprecate every obsolete integration: `transformers` has recently integrated `autoawq` and `autogptq` for instance. ([PR](https://github.com/outlines-dev/outlines/pull/527))
* See if we can integrate to a library that provides state-space models via a logit processing function;
* Integrate with llama.cpp via a logits processor;
* Integrate with exllamav2 via a logits processor;

### Push structured generation further

We're just getting started!

* Improve the performance of existing structured generation algorithms;
* Improve the correctness of structured generation algorithms;
* Add ready-to-use grammars in the [grammars](https://github.com/outlines-dev/grammars) repository or in a submodule in Outlines.

### Keep developing Outlines functions

Functions are awesome, use them!

* Implement a CLI `outlines serve` that allows to serve Outlines functions locally;
* Add more functions to the [functions](https://github.com/outlines-dev/functions) repository.

### Serve structured generation

We want to make it easier to serve structured generation and outlines functions.

* Implement the outlines serve CLI `outlines serve`
  - Serve local APIs that perform structured generation;
  - Serve Outlines functions.

### Improve the generation layer

* Use `transformers`'s private API to prepare inputs for generation inside the `Transformers` class;
* Support successions of model generation and text infilling for methods like Beam Search and SMC;
* Differentiate by adding new caching methods: attention sink, trie-based caching, etc;
* Differentiate by implementing SMC;
* Implement Beam Search;
* Add token healing.

### A more seamless integration with OpenAI

* Provide the same user interface for OpenAI and open source models so they are easily interchangeable;
* Integrate the function calling API.

## Last word

This roadmap was influenced by the expressed interests of the community. If it doesn't reflect your needs please come and [share your experience with us](https://discord.gg/ZxBxyWmW5n).
