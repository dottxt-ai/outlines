# Samplers

## Multinomial sampling

Outlines defaults to the multinomial sampler without top-p or top-k sampling, and temperature equal to 1. Not specifying a sampler is equivalent to:

```python
from outlines import models, generate, samplers


model = models.transformers("mistralai/Mistral-7B-0.1")
sampler = samplers.multinomial()

generator = generate.text(model, sampler)
answer = generator("What is 2+2?")

print(answer)
# 4
```

You can ask the generator to take multiple samples by passing the number of samples when initializing the sampler:

```python
from outlines import models, generate, samplers


model = models.transformers("mistralai/Mistral-7B-0.1")
sampler = samplers.multinomial(3)

generator = generate.text(model, sampler)
answer = generator("What is 2+2?")

print(answer)
# [4, 4, 4]
```

If you ask multiple samples for a batch of prompt the returned array will be of shape `(num_samples, num_batches)`:

```python
from outlines import models, generate, samplers


model = models.transformers("mistralai/Mistral-7B-0.1")
sampler = samplers.multinomial(3)

generator = generate.text(model, sampler)
answer = generator(["What is 2+2?", "What is 3+3?"])

print(answer)
# [[4, 4, 4], [6, 6, 6]]
```


## Greedy sampler

You can also use the greedy sampler. For this you need to initialize the generator with the sampler:


```python
from outlines import models, generate, samplers


model = models.transformers("mistralai/Mistral-7B-0.1")
sampler = samplers.greedy()

generator = generate.text(model, sampler)
answer = generator("What is 2+2?")

print(answer)
# 4
```

You cannot ask for multiple samples with the greedy sampler since it does not clear what the result should be.
