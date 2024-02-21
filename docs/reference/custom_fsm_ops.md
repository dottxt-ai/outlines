# Custom FSM Operations

```RegexFSM.from_interegular_fsm``` leverages the flexibility of ```interegular.FSM``` to use the available operations in ```interegular```.

## Examples

### ```difference```

Returns an FSM which recognises only the strings recognised by the first FSM in the list, but none of the others.

```python
list_of_strings_pattern = """\["[^"\s]*"(?:,"[^"\s]*")*\]"""
pink_elephant_pattern = """.*(pink|elephant).*"""

list_of_strings_fsm = interegular.parse_pattern(list_of_strings_pattern).to_fsm()
pink_elephant_fsm = interegular.parse_pattern(pink_elephant_pattern).to_fsm()

list_of_strings_fsm.accepts('["a","pink","elephant"]')
# True

difference_fsm = list_of_strings_fsm - pink_elephant_fsm

difference_fsm_fsm.accepts('["a","pink","elephant"]')
# False
difference_fsm_fsm.accepts('["a","blue","donkey"]')
# True
```

### ```union```

Returns a finite state machine which accepts any sequence of symbols that is accepted by either self or other.

```python
list_of_strings_pattern = """\["[^"\s]*"(?:,"[^"\s]*")*\]"""
tuple_of_strings_pattern = """\("[^"\s]*"(?:,"[^"\s]*")*\)"""

list_of_strings_fsm = interegular.parse_pattern(list_of_strings_pattern).to_fsm()
tuple_of_strings_fsm = interegular.parse_pattern(tuple_of_strings_pattern).to_fsm()

list_of_strings_fsm.accepts('("a","pink","elephant")')
# False

union_fsm = list_of_strings_fsm|tuple_of_strings_fsm

union_fsm.accepts('["a","pink","elephant"]')
# True
union_fsm.accepts('("a","blue","donkey")')
# True
```

### ```intersection```

Returns an FSM which accepts any sequence of symbols that is accepted by both of the original FSMs.

```python
list_of_strings_pattern = """\["[^"\s]*"(?:,"[^"\s]*")*\]"""
pink_elephant_pattern = """.*(pink|elephant).*"""

list_of_strings_fsm = interegular.parse_pattern(list_of_strings_pattern).to_fsm()
pink_elephant_fsm = interegular.parse_pattern(pink_elephant_pattern).to_fsm()

list_of_strings_fsm.accepts('["a","blue","donkey"]')
# True

intersection_fsm = list_of_strings_fsm & pink_elephant_fsm

intersection_fsm.accepts('["a","pink","elephant"]')
# True
intersection_fsm.accepts('["a","blue","donkey"]')
# False
```

_There are more operations available, we refer to https://github.com/MegaIng/interegular/blob/master/interegular/fsm.py._

# Loading Custom FSM

```python
import outlines

generator = outlines.generate.fsm(model, custom_fsm)

response = generator(prompt)
```
