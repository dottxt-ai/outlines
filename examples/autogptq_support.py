'''
This is a simple example to show how `outlines` can leverage quantized models from Hugging Face Model Hub using `auto-gptq`
'''

## make sure to install the following libraries
# pip install -q outlines transformers accelerate auto-gptq optimum

import outlines.models as models
import outlines.text as text

# loading the quantized open-hermes-2 from HF Model Hub into the GPU 

model = models.transformers("TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ",device="cuda:0")

# define the function whose signature the outlines should learn 

def add(a: int, b: int):
    return a + b

generator = text.generate.json(model, add)

prompt = "Return two integers named a and b respectively. a is odd and b even."

result = generator(prompt)

# print(result) 
# {'a': 1, 'b': 2}

print(add(**result))

# 3 

