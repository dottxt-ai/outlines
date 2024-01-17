import outlines.generate as generate
import outlines.models as models

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

model = models.transformers("hf-internal-testing/tiny-random-gpt2")
batch_size = 10
for grammar in [nlamb_grammar, calc_grammar, dyck_grammar, json_grammar]:
    generator = generate.cfg(model, grammar, max_tokens=model.model.config.n_positions)
    sequences = generator([" "] * batch_size)
    for seq in sequences:
        try:
            parse = generator.fsm.parser.parse(seq)
            assert parse is not None
            print("SUCCESS", seq)
        except Exception:  # will also fail if goes over max_tokens / context window
            print("FAILURE", seq)
