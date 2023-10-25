# Introduction

Here is a simple Outlines program that highlights some of its key features:

```python
import outlines.text as text
import outlines.models as models

@text.prompt
def where_from(expression):
     "What's the origin of '{{ expression }}'?"


complete = models.text_completion.openai("text-davinci-003")

hello_world = where_from("Hello world")
foobar = where_from("Foo Bar")
answer = complete([hello_world, foobar], samples=3, stop_at=["."])
```
