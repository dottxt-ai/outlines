---
title: Welcome to Outlines!
---

Outlines is a Python library that allows you to use Large Language Model in a simple and robust way (with guided generation). It is built by the [.txt](https://dottxt.co), and is already used in production by many companies.

## What models do you support?

We support Openai, but the true power of Outlines is unleashed with Open Source models available via the Transformers, AutoAWQ and AutoGPTQ libraries. If you want to build and maintain an integration with another library, [get in touch][discord].

## What are the main features?

<div class="grid cards" markdown>
-    :material-chat-processing-outline:{ .lg .middle } __Powerful Prompt Templating__

     ---

     Better manage your prompts' complexity with prompt templating

    [:octicons-arrow-right-24: Learn more](reference/prompting.md)


-   :material-regex:{ .lg .middle } __Make LLMs follows a Regex__

    ---

    Generate text that parses correctly 100% of the time

    [:octicons-arrow-right-24: Guide LLMs](reference/regex.md)

-   :material-code-json:{ .lg .middle } __Make LLMs generate valid JSON__

    ---

    No more invalid JSON outputs, 100% guaranteed

    [:octicons-arrow-right-24: Generate JSON](reference/json.md)

-   :material-keyboard-outline:{ .lg .middle } __Rich text generation primitives__

    ---

    Multiple choice, dynamic stopping with OpenAI and Open Source models

    [:octicons-arrow-right-24: Generate text](reference/index.md)

</div>

## Philosphy

**Outlines** 〰 is a library for neural text generation. You can think of it as a
more flexible replacement for the `generate` method in the
[transformers](https://github.com/huggingface/transformers) library.

**Outlines** 〰 helps developers *guide text generation* to build robust
interfaces with external systems. Provides generation methods that
guarantee that the output will match a regular expressions, or follow
a JSON schema.

**Outlines** 〰 provides *robust prompting primitives* that separate the prompting
from the execution logic and lead to simple implementations of few-shot
generations, ReAct, meta-prompting, agents, etc.

**Outlines** 〰 is designed as a *library* that is meant to be compatible the
broader ecosystem, not to replace it. We use as few abstractions as possible,
and generation can be interleaved with control flow, conditionals, custom Python
functions and calls to other libraries.

**Outlines** 〰 is *compatible with all models*. It only interfaces with models
via the next-token logits. It can be used with API-based models as well.


## Why Outlines over alternatives?

Outlines is built at [.txt](https://dottxt.co) by engineers with decades of experience. We do not use unnecessary abstractions that tend to get in your way. We provide guided generation that enable reliable workflows.

## Acknowledgements

<figure markdown>
  <a href="https://www.normalcomputing.ai">
  ![Normal Computing logo](assets/images/normal_computing.jpg){ width="150" }
  </a>
</figure>

Outlines was originally developed at [@NormalComputing](https://twitter.com/NormalComputing) by [@remilouf](https://twitter.com/remilouf) and [@BrandonTWillard](https://twitter.com/BrandonTWillard). It is now maintained by [.txt](https://dottxt.co).

[discord]: https://discord.gg/R9DSu34mGd
