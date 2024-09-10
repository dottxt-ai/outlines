# Samplers

Outlines offers different sequence sampling algorithms, and we will integrate more in the future. You can read [this blog post](https://huggingface.co/blog/how-to-generate) for an overview of the different sampling algorithm.

Samplers provide control over the sampling process, allowing you to influence the output of the model. This can include controlling randomness (temperature), biasing towards certain tokens (top-k, top-p), or sequence generation (beam search).

## Multinomial sampling

Multinomial sampling is the default sampling algorithm in Outlines. You should think of it as calculating the probability distribution over all tokens and then selecting tokens at random according to that distribution.

For example, pretend we have a model with only two possible tokens: "A" and "B". For a fixed prompt such as "Pick A or B" The language model calculates probability for each token:

| Token | Probability |
|-------|-------------|
| A     | 0.75        |
| B     | 0.25        |

You'd expect to receive "A" 75% of the time and "B" 25% of the time.

### Parameters

- `samples`: Number of samples to generate (default: 1)
- `top_k`: Only consider the top k tokens (optional)
- `top_p`: Only consider the top tokens with cumulative probability >= p (optional)
- `temperature`: Controls randomness of sampling (optional)

### Default behavior

Outlines defaults to the multinomial sampler without top-p or top-k sampling, and temperature equal to 1. 

Not specifying a sampler is equivalent to:

```python
from outlines import models, generate, samplers


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
sampler = samplers.multinomial()

generator = generate.text(model, sampler)
answer = generator("What is 2+2?")

print(answer)
# 4
```

### Batching

You can ask the generator to take multiple samples by passing the number of samples when initializing the sampler:

```python
from outlines import models, generate, samplers


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
sampler = samplers.multinomial(3)

generator = generate.text(model, sampler)
answer = generator("What is 2+2?")

print(answer)
# [4, 4, 4]
```

If you ask multiple samples for a batch of prompts the returned array will be of shape `(num_samples, num_batches)`:

```python
from outlines import models, generate, samplers


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
sampler = samplers.multinomial(3)

generator = generate.text(model, sampler)
answer = generator(["What is 2+2?", "What is 3+3?"])

print(answer)
# [[4, 4, 4], [6, 6, 6]]
```

### Temperature

You can control the temperature with

```python
from outlines import models, generate, samplers


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
sampler = samplers.multinomial(3, temperature=0.5)

generator = generate.text(model, sampler)
answer = generator(["What is 2+2?", "What is 3+3?"])

print(answer)
```

If you would like to use `temperature=0.0`, please use `sampler=samplers.greedy()` instead.

### Top-k sampling

You can ask Outlines to only consider the top-k logits at each step by specifying the value of the `top-k` keyword argument when initializing the sampler.

```python
sampler = samplers.multinomial(3, top_k=10)
```

### Top-p sampling

You can ask Outlines to only consider the highest probability tokens such that their cumulative probability is greater than a threshold `p`. Specify the `top_p` keyword argument when initializing the sampler:


```python
sampler = samplers.multinomial(3, top_p=0.95)
```

## Greedy sampler

Greedy sampling selects the token with the highest probability at each step. It's deterministic and always produces the same output for a given input.

To use the greedy sampler, initialize the generator with the sampler:


```python
from outlines import models, generate, samplers


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
sampler = samplers.greedy()

generator = generate.text(model, sampler)
answer = generator("What is 2+2?")

print(answer)
# 4
```

You cannot ask for multiple samples with the greedy sampler since it does not clear what the result should be. Only the most likely token can be returned.


## Beam Search

Beam search maintains multiple candidate sequences at each step, potentially finding better overall sequences than greedy or multinomial sampling.

To use Beam Search, initialize the generator with the sampler:

```python
from outlines import models, generate, samplers


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
sampler = samplers.beam_search(beams=5)

generator = generate.text(model, sampler)
answer = generator("What is 2+2?")

print(answer)
# 4
```
!!! Warning "Compatibility"

    Only models from the `transformers`  and `exllamav2 ` libraries are compatible with Beam Search.

### Parameters

- `beams`: Number of beams to use (default: 1)

