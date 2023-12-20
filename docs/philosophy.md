---
title: Philosophy
icon: material/thought-bubble
---

# Outlines' philosphy

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
