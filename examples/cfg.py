from lark.exceptions import UnexpectedCharacters, UnexpectedToken

import outlines.generate as generate
import outlines.models as models

# examples from https://lark-parser.readthedocs.io/en/latest/examples/index.html

nlamb_grammar = """
    start: sentence

    sentence: noun verb noun        -> simple
            | noun verb "like" noun -> comparative

    noun: adj? NOUN
    verb: VERB
    adj: ADJ

    NOUN: "flies" | "bananas" | "fruit"
    VERB: "like" | "flies"
    ADJ: "fruit"

    %import common.WS
    %ignore WS
"""

calc_grammar = """
    ?start: sum
          | NAME "=" sum    -> assign_var

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: atom
        | product "*" atom  -> mul
        | product "/" atom  -> div

    ?atom: NUMBER           -> number
         | "-" atom         -> neg
         | NAME             -> var
         | "(" sum ")"

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS_INLINE

    %ignore WS_INLINE
"""

model = models.transformers("hf-internal-testing/tiny-random-gpt2")
batch_size = 10
max_tokens = 30
for grammar in [nlamb_grammar, calc_grammar]:
    generator = generate.cfg(model, grammar)
    sequences = generator([" "] * batch_size, max_tokens=max_tokens)
    for seq in sequences:
        try:
            parse = generator.fsm.parser.parse(seq)
            assert parse is not None
            print("SUCCESS", seq)
        except (UnexpectedCharacters, UnexpectedToken):
            if generator.fsm.num_tokens_generated == max_tokens:
                print("MAXTOKEN", seq)
            else:
                print("FAILURE", seq)
        except Exception:
            print("FAILURE", seq)
