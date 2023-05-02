"""ReAct

This example was inspired by the LQML library [1]_. The ReAct framework was
first developed in [2]_ and augments Chain-of-Thought prompting with the ability
for the model to query external sources.

References
----------
.. [1] Beurer-Kellner, L., Fischer, M., & Vechev, M. (2022). Prompting Is Programming: A Query Language For Large Language Models. arXiv preprint arXiv:2212.06094.
.. [2] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

"""
import requests  # type: ignore

import outlines
import outlines.models as models
import outlines.text as text

outlines.cache.disable()


@text.prompt
def build_reAct_prompt(question):
    """What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
    Tho 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado ...
    Act 2: Search 'Colorado orogeny'
    Obs 2: The Colorado orogeny was an episode of mountain building (an orogeny) ...
    Tho 3: It does not mention the eastern sector. So I need to look up eastern sector.
    ...
    Tho 4: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
    Act 5: Finish '1,800 to 7,000 ft'
    {{ question }}
    """


@text.prompt
def add_mode(i, mode, result, prompt):
    """{{ prompt }}
    {{ mode }} {{ i }}: {{ result }}
    """


def search_wikipedia(query: str):
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles={query}&origin=*"
    response = requests.get(url)
    page = response.json()["query"]["pages"]
    return ".".join(list(page.values())[0]["extract"].split(".")[:2])


mode_model = models.text_completion.openai(
    "gpt-3.5-turbo", is_in=["Tho", "Act"], max_tokens=2
)
action_model = models.text_completion.openai(
    "text-davinci-003", is_in=["Search", "Finish"], max_tokens=2
)
thought_model = models.text_completion.openai(
    "text-davinci-003", stop_at=["\n"], max_tokens=128
)
subject_model = models.text_completion.openai(
    "text-davinci-003", stop_at=["'"], max_tokens=128
)

prompt = build_reAct_prompt("Where is Apple Computers headquarted? ")

for i in range(1, 10):
    mode = mode_model(prompt)
    if mode == "Tho":
        prompt = add_mode(i, mode, "", prompt)
        thought = thought_model(prompt)
        prompt += f"{thought}"
    if mode == "Act":
        prompt = add_mode(i, mode, "", prompt)
        action = action_model(prompt)
        prompt += f"{action} '"
        subject = " ".join(subject_model(prompt).split()[:2])
        prompt += f"{subject}'"
        if action == "Search":
            result = search_wikipedia(subject)
            prompt = add_mode(i, "Obs", result, prompt)
        else:
            break

print(prompt)
