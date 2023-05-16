"""Meta-prompting examples.

References
----------

.. [0] "Prompting is programming: A Query Language for Large Language Models"
       https://arxiv.org/abs/2212.06094
.. [1] "Prompt programming For Large Language Models: Beyond the Few-Shot Paradigm"
       https://arxiv.org/abs/2102.07350.

"""
import argparse

import outlines.models as models
import outlines.text as text


def split_into_steps(question, model_name: str):
    @text.prompt
    def solve(question):
        """{{question}}
        Let's solve this problem by splitting it into steps.
        """

    complete = models.text_completion.openai(model_name)

    prompt = solve(question)
    answer = complete(prompt)
    completed = prompt + answer

    return completed


def fill_in_the_blanks(question, model_name: str):
    @text.prompt
    def determine_goal(question):
        """{{question}}

        In order to solve this problem, we will analyze each of the options and determine
        """

    @text.prompt
    def solve(memory):
        """{{memory}}. Let's begin."""

    complete = models.text_completion.openai(model_name)

    prompt = determine_goal(question)
    answer = complete(prompt, stop_at=["."])
    prompt = solve(prompt + answer)
    answer = complete(prompt, stop_at=["."])
    completed = prompt + answer

    return completed


def ask_an_expert(question, model_name: str):
    @text.prompt
    def find_expert(question):
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

    @text.prompt
    def get_answer(question, expert, memory):
        """
        {{memory}}
        I am ready to ask my question.
        "{{expert}}" I say,
        {{question}}
        """

    complete_expert = models.text_completion.openai(model_name)
    complete_answer = models.text_completion.openai(model_name)

    prompt = find_expert(question)
    expert = complete_expert(prompt, stop_at=['"'])
    prompt = get_answer(question, expert, prompt + expert)
    answer = complete_answer(prompt)
    completed = prompt + answer

    return completed


def ask_an_expert_simple(question, model_name: str):
    @text.prompt
    def find_expert(question):
        """
        Q: {{question}}
        A: A good person to answer this question would be
        """

    @text.prompt
    def get_answer(expert, memory):
        """
        {{memory}}.

        For instance, {{expert}} would answer
        """

    model_expert = models.text_completion.openai(model_name)
    model_answer = models.text_completion.openai(model_name)

    prompt = find_expert(question)
    expert = model_expert(prompt, stop_at=["\n", "."])
    prompt = get_answer(expert, prompt + expert)
    answer = model_answer(prompt)
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
        default="gpt-3.5-turbo",
        help="The Large Language Model to use to run the examples.",
    )
    args = parser.parse_args()

    math_q = "f(x) = x*x. What is f(f(3))?"
    sat_q = """

Directions: In the following question, a related pair of words or phrases \
is followed by five pairs of words or phrases. Choose the pair that best \
expresses a relationship similar to that in the original pair. \

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
    run_example(split_into_steps, sat_q, args.model)
    run_example(fill_in_the_blanks, sat_q, args.model)
    run_example(ask_an_expert, alignment_q, args.model)
    run_example(ask_an_expert_simple, meaning_q, args.model)
