# Multiple choices

Choice between different options
In some cases we know the output is to be chosen between different options. We can restrict the completion’s output to these choices using the is_in keyword argument:

```python
import outlines.models as models

complete = models.openai("gpt-3.5-turbo")
answer = complete(
    "Pick the odd word out: skirt, dress, pen, jacket",
    is_in=["skirt", "dress", "pen", "jacket"]
)
```
