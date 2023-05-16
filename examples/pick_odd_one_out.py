"""Chain-of-thought prompting for Odd one out classification.

Example taken from the LQML library [1]_.

References
----------
.. [1] Beurer-Kellner, L., Fischer, M., & Vechev, M. (2022).
       Prompting Is Programming: A Query Language For Large Language Models.
       arXiv preprint arXiv:2212.06094.

"""
import outlines.models as models
import outlines.text as text


@text.prompt
def build_ooo_prompt(options):
    """
    Pick the odd word out: skirt, dress, pen, jacket.
    skirt is clothing, dress is clothing, pen is an object, jacket is clothing.
    So the odd one is pen.

    Pick the odd word out: Spain, France, German, England, Singapore.
    Spain is a country, France is a country, German is a language, ...
    So the odd one is German.

    Pick the odd word out: {{ options | join(", ") }}.

    """


reasoning_model = models.text_completion.openai("text-davinci-003")
result_model = models.text_completion.openai("text-davinci-003")

options = ["sea", "mountains", "plains", "sock"]
prompt = build_ooo_prompt(options)
reasoning = reasoning_model(prompt, stop_at=["Pick the odd word", "So the odd one"])
prompt += reasoning
result = result_model(prompt)
prompt += result
print(prompt)
