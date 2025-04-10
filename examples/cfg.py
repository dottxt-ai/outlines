from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines
from outlines.types import CFG


nlamb_grammar = r"""
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

calc_grammar = r"""
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

    %import common.LETTER -> NAME
    %import common.INT -> NUMBER
    %import common.WS_INLINE

    %ignore WS_INLINE
"""

dyck_grammar = r"""
    start: s
    s: /a+/
    | "(" s ")"
    | "{" s "}"
    | "[" s "]"
"""

json_grammar = r"""
    ?start: value

    ?value: object
          | array
          | string
          | SIGNED_NUMBER      -> number
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null

    array  : "[" [value ("," value)*] "]"
    object : "{" [pair ("," pair)*] "}"
    pair   : string ":" value

    inner: /([^"]|\\\")+/ |
    string : "\"" inner "\""

    %import common.SIGNED_NUMBER
    %import common.WS

    %ignore WS
"""

model_name = "hf-internal-testing/tiny-random-gpt2"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name),
)

batch_size = 10
for grammar in [nlamb_grammar, calc_grammar, dyck_grammar, json_grammar]:
    generator = outlines.Generator(model, CFG(grammar))
    sequences = generator([" "] * batch_size, max_tokens=model.model.config.n_positions)
    for seq in sequences:
        try:
            parse = generator.logits_processor.guide.parser.parse(seq)
            assert parse is not None
            print("SUCCESS", seq)
        except Exception:  # will also fail if goes over max_new_tokens / context window
            print("FAILURE", seq)
