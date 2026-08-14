"""Chain-of-thought prompting for Odd one out classification.

Example taken from the LQML library [1]_.

References
----------
.. [1] Beurer-Kellner, L., Fischer, M., & Vechev, M. (2022).
       Prompting Is Programming: A Query Language For Large Language Models.
       arXiv preprint arXiv:2212.06094.

"""

import json

import openai

import outlines
from outlines import Generator
from outlines.types import JsonSchema


build_ooo_prompt = outlines.Template.from_file("prompts/pick_odd_one_out.txt")

options = ["sea", "mountains", "plains", "sock"]
options_schema = JsonSchema({
    "type": "object",
    "properties": {
        "result": {
            "type": "string",
            "enum": options
        }
    },
    "required": ["result"]
})

model = outlines.from_openai(openai.OpenAI(), "gpt-4o-mini")
gen_text = Generator(model)
gen_choice = Generator(model, options_schema)

prompt = build_ooo_prompt(options=options)
reasoning = gen_text(prompt, stop=["Pick the odd word", "So the odd one"])
prompt += reasoning
raw_result = gen_choice(prompt)
result = json.loads(raw_result)["result"]
prompt += result
print(result)
