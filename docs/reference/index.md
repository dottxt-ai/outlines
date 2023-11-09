# Reference

## Constrained generation

While LLM capabilities are increasingly impressive, we can make their output more reliable by steering the generation. Outlines thus offers mechanisms to specify high level constraints on text completions by generative language models.

Stopping sequence
By default, language models stop generating tokens after and <EOS> token was generated, or after a set maximum number of tokens. Their output can be verbose, and for practical purposes it is often necessary to stop the generation after a given sequence has been found instead. You can use the stop_at keyword argument when calling the model with a prompt:

```python
import outlines.models as models

complete = models.text_completion.openai("text-davinci-002")
expert = complete("Name an expert in quantum gravity.", stop_at=["\n", "."])
```
