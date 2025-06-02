# Custom FSM Operations

Outlines is fast because it compiles regular expressions into an index ahead of inference. To do so we use the equivalence between regular expressions and Finite State Machines (FSMs), and the library [interegular](https://github.com/MegaIng/interegular) to perform the translation.

Alternatively, one can pass a FSM built using `integular` directly to structure the generation.

## Example

### Using the `difference` operation

In the following example we build a fsm which recognizes only the strings valid to the first regular expression but not the second. In particular, it will prevent the words "pink" and "elephant" from being generated:

```python
import interegular
from outlines import from_transformers, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer

list_of_strings_pattern = """\["[^"\s]*"(?:,"[^"\s]*")*\]"""
pink_elephant_pattern = """.*(pink|elephant).*"""

list_of_strings_fsm = interegular.parse_pattern(list_of_strings_pattern).to_fsm()
pink_elephant_fsm = interegular.parse_pattern(pink_elephant_pattern).to_fsm()

difference_fsm = list_of_strings_fsm - pink_elephant_fsm

difference_fsm_fsm.accepts('["a","pink","elephant"]')
# False
difference_fsm_fsm.accepts('["a","blue","donkey"]')
# True

model_name = "microsoft/Phi-3-mini-4k-instruct"
model = from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

generator = Generator(model, difference_fsm)
response = generator("Don't talk about pink elephants")
```

To see the other operations available, consult [interegular's documentation](https://github.com/MegaIng/interegular/blob/master/interegular/fsm.py).
