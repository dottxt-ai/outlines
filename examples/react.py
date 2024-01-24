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


@outlines.prompt
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


@outlines.prompt
def add_mode(i, mode, result, prompt):
    """{{ prompt }}
    {{ mode }} {{ i }}: {{ result }}
    """


def search_wikipedia(query: str):
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles={query}&origin=*"
    response = requests.get(url)
    page = response.json()["query"]["pages"]
    return ".".join(list(page.values())[0]["extract"].split(".")[:2])


prompt = build_reAct_prompt("Where is Apple Computers headquarted? ")
model = models.openai("gpt-3.5-turbo")
complete = outlines.generate.text(model)

for i in range(1, 10):
    mode = complete.generate_choice(prompt, choices=["Tho", "Act"], max_tokens=128)
    prompt = add_mode(i, mode, "", prompt)

    if mode == "Tho":
        thought = complete(prompt, stop_at="\n", max_tokens=128)
        prompt += f"{thought}"
    elif mode == "Act":
        action = complete.generate_choice(
            prompt, choices=["Search", "Finish"], max_tokens=128
        )
        prompt += f"{action} '"

        subject = complete(
            prompt, stop_at=["'"], max_tokens=128
        )  # Apple Computers headquartered
        subject = " ".join(subject.split()[:2])
        prompt += f"{subject}'"

        if action == "Search":
            result = search_wikipedia(subject)
            prompt = add_mode(i, "Obs", result, prompt)
        else:
            break

print(prompt)
