# Custom types

Outlines provides custom Pydantic types so you can focus on your use case rather than on writing regular expressions:

- Using `outlines.types.ZipCode` will generate valid US Zip(+4) codes.
- Using `outlines.types.PhoneNumber` will generate valid US phone numbers.

You can use these types in Pydantic schemas for JSON-structured generation:

```python
from pydantic import BaseModel

from outlines import models, generate, types


class Client(BaseModel):
    name: str
    phone_number: types.PhoneNumber
    zip_code: types.ZipCode


model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.json(model, Client)
result = generator(
    "Create a client profile with the fields name, phone_number and zip_code"
)
print(result)
# name='Tommy' phone_number='129-896-5501' zip_code='50766'
```

Or simply with `outlines.generate.format`:

```python
from pydantic import BaseModel

from outlines import models, generate, types


model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.format(model, types.PhoneNumber)
result = generator(
    "Return a US Phone number: "
)
print(result)
# 334-253-2630
```


We plan on adding many more custom types. If you have found yourself writing regular expressions to generate fields of a given type, or if you could benefit from more specific types don't hesite to [submit a PR](https://github.com/outlines-dev/outlines/pulls) or [open an issue](https://github.com/outlines-dev/outlines/issues/new/choose).
