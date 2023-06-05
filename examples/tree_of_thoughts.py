"""Tree of Thoughts

This is a simple reimplementation of the Tree of Thoughts [0]_ for solving "Game of 24" 
using outlines for prompt management.

References
----------

.. [0] "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
       https://arxiv.org/abs/2305.10601

"""
import re
from collections import deque
import backoff
import numpy as np
import openai
import outlines.models as models
import outlines.text as text
import sympy
from tqdm import tqdm

cot_examples = [
    {
        "input": "4 4 6 8",
        "steps": [
            "4 + 8 = 12 (left: 4 6 12)",
            "6 - 4 = 2 (left: 2 12)",
            "2 * 12 = 24 (left: 24)",
        ],
        "output": "(6 - 4) * (4 + 8) = 24",
    },
    {
        "input": "2 9 10 12",
        "steps": [
            "12 * 2 = 24 (left: 9 10 24)",
            "10 - 9 = 1 (left: 1 24)",
            "24 * 1 = 24 (left: 24)",
        ],
        "output": "(12 * 2) * (10 - 9) = 24",
    },
    {
        "input": "4 9 10 13",
        "steps": [
            "13 - 10 = 3 (left: 3 4 9)",
            "9 - 3 = 6 (left: 4 6)",
            "4 * 6 = 24 (left: 24)",
        ],
        "output": "4 * (9 - (13 - 10)) = 24",
    },
    {
        "input": "1 4 8 8",
        "steps": [
            "8 / 4 = 2 (left: 1 2 8)",
            "1 + 2 = 3 (left: 3 8)",
            "3 * 8 = 24 (left: 24)",
        ],
        "output": "(1 + 8 / 4) * 8 = 24",
    },
    {
        "input": "5 5 5 9",
        "steps": [
            "5 + 5 = 10 (left: 5 9 10)",
            "10 + 5 = 15 (left: 9 15)",
            "15 + 9 = 24 (left: 24)",
        ],
        "output": "((5 + 5) + 5) + 9 = 24",
    },
]


@text.prompt
def cot_prompt(input, examples=cot_examples):
    """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
    {% for example in examples %}
    Input: {{ example.input }}
    Steps:
    {% for step in example.steps %}
    {{ step }}
    {% endfor %}
    Answer: {{ example.output }}
    {% endfor %}
    Input: {{input}}
    """


@backoff.on_exception(
    backoff.expo,
    (OSError, openai.error.RateLimitError, openai.error.OpenAIError),
)
def model(*args, **kwargs):
    # gpt = models.text_completion.openai("gpt-4", max_tokens = 1000, temperature = 0.7)
    gpt = models.text_completion.openai(
        "gpt-3.5-turbo", max_tokens=1000, temperature=0.7
    )

    return gpt(*args, **kwargs)


#################################
#######  TREE OF THOUGHTS  ######
#################################

## proposals and next_steps are the same thing
propose_examples = [
    {
        "input": "2 8 8 14",
        "next_steps": [
            "2 + 8 = 10 (left: 8 10 14)",
            "8 / 2 = 4 (left: 4 8 14)",
            "14 + 2 = 16 (left: 8 8 16)",
            "2 * 8 = 16 (left: 8 14 16)",
            "8 - 2 = 6 (left: 6 8 14)",
            "14 - 8 = 6 (left: 2 6 8)",
            "14 /  2 = 7 (left: 7 8 8)",
            "14 - 2 = 12 (left: 8 8 12)",
        ],
    }
]


@text.prompt
def propose_prompt(input, examples=propose_examples):
    """
    {% for example in examples %}
    Input: {{ example.input }}
    Possible next steps:
    {% for next_step in example.next_steps %}
    {{ next_step }}
    {% endfor %}
    {% endfor %}
    Input: {{ input }}
    Possible next steps:
    """


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split("\n")[-1]
    return last_line.split("left: ")[-1].split(")")[0]


def get_proposals(x: str, y: str = "") -> list[str]:
    current_numbers = get_current_numbers(y if y else x)
    if current_numbers == "24":
        # generate full template, final stage
        prompt = cot_prompt(input=x) + "Steps:" + y
    else:
        prompt = propose_prompt(current_numbers)

    proposals_str = model(prompt)
    proposals_arr = proposals_str.split("\n")
    proposals_arr = [y + _ + "\n" for _ in proposals_arr]  # past steps also included
    return proposals_arr


value_examples = [
    {"input": "10 14", "steps": ["10 + 14 = 24"], "output": "sure"},
    {
        "input": "11 12",
        "steps": ["11 + 12 = 23", "12 - 11 = 1", "11 * 12 = 132", "11 / 12 = 0.91"],
        "output": "impossible",
    },
    {
        "input": "4 4 10",
        "steps": [
            "4 + 4 + 10 = 8 + 10 = 18",
            "4 * 10 - 4 = 40 - 4 = 36",
            "(10 - 4) * 4 = 6 * 4 = 24",
        ],
        "output": "sure",
    },
    {"input": "4 9 11", "steps": ["9 + 11 + 4 = 20 + 4 = 24"], "output": "sure"},
    {
        "input": "5 7 8",
        "steps": [
            "5 + 7 + 8 = 12 + 8 = 20",
            "(8 - 5) * 7 = 3 * 7 = 21",
            "I cannot obtain 24 now, but numbers are within a reasonable range",
        ],
        "output": "likely",
    },
    {
        "input": "5 6 6",
        "steps": [
            "5 + 6 + 6 = 17",
            "(6 - 5) * 6 = 1 * 6 = 6",
            "I cannot obtain 24 now, but numbers are within a reasonable range",
        ],
        "output": "likely",
    },
    {
        "input": "10 10 11",
        "steps": [
            "10 + 10 + 11 = 31",
            "(11 - 10) * 10 = 10",
            "10 10 10 are all too big",
        ],
        "output": "impossible",
    },
    {
        "input": "1 3 3",
        "steps": ["1 * 3 * 3 = 9", "(1 + 3) * 3 = 12", "1 3 3 are all too small"],
        "output": "impossible",
    },
    {"input": "24", "steps": ["24 = 24 (solved, no steps needed)"], "output": "sure"},
]

# this can be more descriptive i feel, difficult to train using 1-line input and expecting it do for previous states
value_last_step_examples = [
    {"input": "4 4 6 8", "answer": "(4 + 8) * (6 - 4) = 24", "judge": "sure"},
    {"input": "2 9 10 12", "answer": "2 * 12 * (10 - 9) = 24", "judge": "sure"},
    {"input": "4 9 10 13", "answer": "(13 - 9) * (10 - 4) = 24", "judge": "sure"},
    {"input": "4 4 6 8", "answer": "(4 + 8) * (6 - 4) + 1 = 25", "judge": "impossible"},
    {"input": "2 9 10 12", "answer": "2 * (12 - 10) = 24", "judge": "impossible"},
    {"input": "4 9 10 13", "answer": "(13 - 4) * (10 - 9) = 24", "judge": "impossible"},
]


@text.prompt
def value_prompt(input, examples=value_examples):
    """Evaluate if given numbers can reach 24 (sure/likely/impossible)
    {% for example in examples %}
    Input: {{ example.input }}
    {% for step in example.steps %}
    {{ step }}
    {% endfor %}
    {{ example.output }}
    {% endfor %}
    Input: {{input}}
    """


@text.prompt
def value_last_step_prompt(input, answer, examples=value_last_step_examples):
    """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
    {% for example in examples %}
    Input: {{ example.input }}
    Answer: {{ example.answer }}
    Judge: {{ example.judge }}
    {% endfor %}
    Input: {{input}}
    Answer: {{answer}}
    Judge:"""


def get_value_prompt(x, y):
    """
    If final answer is found, we use a different value prompt
    If not, we use another value prompt
    """
    last_line = y.strip().split("\n")[-1]
    if "left: " not in last_line:  # last step
        ans = last_line.lower().replace("answer: ", "")
        return value_last_step_prompt(input=x, answer=ans)
    current_numbers = get_current_numbers(y)
    return value_prompt(current_numbers)


def parse_and_compute_value(x: str, y: str, value_outputs: list[str]) -> float:
    """
    The idea is to determine a value for the output. now we only score the output label.
    we can additionally score the reasoning steps too.
    this calculation need not be perfect
    """
    if len(y.strip().split("\n")) == 4 and "answer" not in y.lower():
        # print(f'4 steps done but no answer found {y=}')
        return 0
    value_names = [_.split("\n")[-1] for _ in value_outputs]
    value_map = {"impossible": 0.001, "likely": 1, "sure": 20}  # TODO: ad hoc
    value = sum(value * value_names.count(name) for name, value in value_map.items())
    return value


def get_value(x, y, n_eval):
    """
    Obtain the value of the partial/completed output y, given the input x.
    Value can be assigned by a LLM or a defined function, or a combination of both for various steps
    Idea is to automate all the steps by a LLM
    """
    value_prompt = get_value_prompt(x, y)
    value_outputs = model(value_prompt, samples=n_eval)
    value = parse_and_compute_value(x, y, value_outputs)
    return value


def tree_of_thoughts(x: str) -> list[str]:
    """
    BFS + Beam search of n_best width and n_steps depth
    """
    n_best = 5  # how many best candidates to select for the next bfs queue
    n_steps = 4  # max 4 steps for this problem (4 numbers)
    n_eval = 3  # how many samples to calculate value

    queue = deque([""])  # current output candidates
    for _ in range(n_steps):
        # 1. generate proposals/next_steps
        proposals = []
        for y in queue:
            y_new = get_proposals(x, y)
            proposals.extend(y_new)

        # 2. score the proposals, and select n_best highest
        values = []
        for y in proposals:
            value = get_value(x, y, n_eval)
            values.append(value)

        sorted_indices = np.argsort(values)[::-1]
        sorted_proposals = [proposals[i] for i in sorted_indices]
        queue = sorted_proposals[:n_best]

    # return many final thoughts (complete outputs)
    return queue


def eval_output_game24(input: str, output: str) -> bool:
    """
    Use regex to check if the output is correct for the given input
    """
    expression = (
        output.strip().split("\n")[-1].lower().replace("answer: ", "").split("=")[0]
    )
    numbers = re.findall(r"\d+", expression)
    problem_numbers = re.findall(r"\d+", input)
    if sorted(numbers) != sorted(problem_numbers):
        return False
    try:
        # print(sympy.simplify(expression))
        return sympy.simplify(expression) == 24
    except Exception as e:
        # print(e)
        return False


def run_tot_bfs():
    """
    This is actually a BEAM search, with beam width as n_best
    """
    # generate for all inputs
    outputs = []
    for x in tqdm(inputs):
        ys = tree_of_thoughts(x)
        outputs.append(ys)

    # check accuracy of the answers
    status_matrix = []
    for x, ys in zip(inputs, outputs):
        status = [eval_output_game24(x, y) for y in ys]
        status_matrix.append(status)

    n_any_correct = np.sum(np.any(status_matrix, axis=1))
    avg_correct = np.mean(status_matrix)

    # print results
    print("ToT Prompting")
    print(f"n_inputs: {len(inputs)}")
    print(f"n_inputs (min 1 correct): {n_any_correct}")
    print(f"avg number of correct answers per input: {avg_correct:.2f}")


if __name__ == "__main__":
    inputs = [
        "1 1 2 8",
        "1 1 4 8",
        "1 1 5 8",
        "4 6 11 11",
        "1 1 3 12"
    ]
    run_tot_bfs()
