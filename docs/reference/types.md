# Type constraints

We can ask completions to be restricted to valid integers or floating-point numbers using the `type` keyword argument, respectively with the “int” or “float” value:

```python
import outlines.models as models

complete = models.openai("gpt-3.5-turbo")
answer = complete(
    "When I was 6 my sister was half my age. Now I’m 70 how old is my sister?",
    type="int"
)
```
