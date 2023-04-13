"""Example from https://dust.tt/spolu/a/d12ac33169"""
import outlines.text as text

examples = [
    {"question": "What is 37593 * 67?", "code": "37593 * 67"},
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "code": "(16-3-4)*2",
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "code": " 2 + 2/2",
    },
]


@text.completion("openai/text-davinci-003", stop_at=["QUESTION"])
def answer_with_code(question, examples):
    """
    {% for example in examples %}
    QUESTION: {{example.question}}
    CODE: {{example.code}}

    {% endfor %}
    QUESTION: {{question}}
    CODE:"""


def execute_code(code):
    result = eval(code)
    return result


question = "Carla is downloading a 200 GB file. She can download 2 GB/minute, but 40% of the way through the download, the download fails. Then Carla has to restart the download from the beginning. How load did it take her to download the file in minutes?"
result_code, _ = answer_with_code(question, examples)
result = execute_code(result_code)
print(result)
