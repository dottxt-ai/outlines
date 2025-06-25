"""Meta-prompting examples.

References
----------

.. [0] "Prompting is programming: A Query Language for Large Language Models"
       https://arxiv.org/abs/2212.06094
.. [1] "Prompt programming For Large Language Models: Beyond the Few-Shot Paradigm"
       https://arxiv.org/abs/2102.07350.

"""

import argparse

import openai

import outlines
from outlines import Template


client = openai.OpenAI()


def split_into_steps(question, model_name: str):
    solve = Template.from_string(
        """{{question}}
        Rephrase : : as a true or false statement, identify an Object, relationship and subject
        """
    )

    model = outlines.from_openai(client, model_name)

    prompt = solve(question=question)
    answer = model(prompt, max_tokens=500)
    prompt += (
        answer
        + "\n what is the only option that displays the same type of relationship as : :?"
    )
    answer = model(prompt, max_tokens=500)
    completed = prompt + answer

    return completed


def fill_in_the_blanks(question, model_name: str):
    determine_goal = Template.from_string(
        """{{question}}

        In order to solve this problem, we will analyze each of the options and determine
        """
    )

    solve = Template.from_string("""{{memory}}. Let's begin.""")

    model = outlines.from_openai(client, model_name)

    prompt = determine_goal(question=question)
    answer = model(prompt, stop=["."])
    prompt = solve(memory=prompt + answer)
    answer = model(prompt, max_tokens=500)
    completed = prompt + answer

    return completed


def ask_an_expert(question, model_name: str):
    find_expert = Template.from_string(
        """
        {{question}}
        I entered my question into the Expert Generator \
        and waited. The Expert Generator will render a \
        simulation of an expert to answer my question. \
        The expert could be anyone, dead or alive, real \
        or fictional; the machine will find the person \
        most qualified to answer the question. For this \
        question in particular, the expert must be someone \
        who has thought a lot about the problem of \
        artificial intelligence and its alignment. \
        The Expert Generator beeped, indicating that it has \
        found the most qualified expert. The name displayed \
        on the screen: "
        """
    )

    get_answer = Template.from_string(
        """
        {{memory}}".
        I am ready to ask my question.
        "{{expert}}" I say,
        {{question}}
        """
    )

    model = outlines.from_openai(client, model_name)

    prompt = find_expert(question=question)
    expert = model(prompt, stop=['"'])
    prompt = get_answer(question=question, expert=expert, memory=prompt+expert)
    answer = model(prompt, max_tokens=500)
    completed = prompt + answer

    return completed


def ask_an_expert_simple(question, model_name: str):
    find_expert = Template.from_string(
        """
        Q: {{question}}
        A: A good person to answer this question would be
        """
    )

    get_answer = Template.from_string(
        """
        {{memory}}.

        For instance, {{expert}} would answer
        """
    )

    model = outlines.from_openai(client, model_name)

    prompt = find_expert(question=question)
    expert = model(prompt, stop=["\n", "."])
    prompt = get_answer(expert=expert, memory=prompt+expert)
    answer = model(prompt, max_tokens=500)
    completed = prompt + answer

    return completed


def run_example(model_fn, question, model_name):
    completed = model_fn(question, model_name)
    print("\n-----------------------")
    print(f"{completed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Meta Prompting examples")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="The Large Language Model to use to run the examples.",
    )
    args = parser.parse_args()

    math_q = "f(x) = x*x. What is f(f(3))?"
    sat_q = """

BRAGGART :: MODESTY
A) FLEDGLING : EXPERIENCE
B) EMBEZZLER : GREED
C) WALLFLOWER : TIMIDITY
D) INVALID : MALADY
E) CANDIDATE : AMBITION

    """
    alignment_q = "What should humankind do to ensure that artificial general intelligence is aligned?"
    meaning_q = "What is the meaning of life?"

    run_example(split_into_steps, math_q, args.model)
    run_example(
        split_into_steps, sat_q.lower(), args.model
    )  # gpt>3.5 usually gets this one right
    run_example(fill_in_the_blanks, sat_q, args.model)
    run_example(ask_an_expert, alignment_q, args.model)
    run_example(ask_an_expert_simple, meaning_q, args.model)
