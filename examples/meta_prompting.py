"""Meta-prompting examples.

References
----------

.. [0] "Prompting is programming: A Query Language for Large Language Models"
       https://arxiv.org/abs/2212.06094
.. [1] "Prompt programming For Large Language Models: Beyond the Few-Shot Paradigm"
       https://arxiv.org/abs/2102.07350.

"""
import argparse

import outlines
from outlines import compose
from outlines.text.models.openai import OpenAI


def split_into_steps(question, model: str):
    prompt = compose(
        """
        ${question}
        Let's solve this problem by splitting it into steps.
        """,
        question=question,
    )
    answer = OpenAI(model)(prompt)

    return prompt, answer


def fill_in_the_blanks(question, model: str):
    meta_prompt = compose(
        """
        ${question}

        In order to solve this problem, we will analyze each of the options and determine
        """,
        question=question,
    )
    goal = OpenAI(model, stops_at=["."])(meta_prompt)

    prompt = compose(
        """
        ${meta_prompt}${goal}. Let's begin.
        """,
        meta_prompt=meta_prompt,
        goal=goal,
    )
    answer = OpenAI(model)(prompt)

    return prompt, answer


def ask_an_expert(question, model: str):
    meta_prompt = compose(
        """
        ${question}
        I entered my question into the Expert Generator
        and waited. The Expert Generator will render a
        simulation of an expert to answer my question.
        The expert could be anyone, dead or alive, real
        or fictional; the machine will find the person
        most qualified to answer the question. For this
        question in particular, the expert must be someone
        who has thought a lot about the problem of
        artificial intelligence and its alignment.
        The Expert Generator beeped, indicating that it has
        found the most qualified expert. The name displayed
        on the screen: "
        """,
        question=question,
    )
    expert = OpenAI(model, stops_at=['"'])(meta_prompt)

    prompt = compose(
        """
        ${prompt}${expert}"
        I am ready to ask my question.
        "${expert}" I say,
        ${question}
        """,
        prompt=meta_prompt,
        expert=expert,
        question=question,
    )
    answer = OpenAI(model)(prompt)
    return prompt, answer


def ask_an_expert_simple(question, model: str):
    meta_prompt = compose(
        """
        Q: ${question}
        A: A good person to answer this question would be
        """,
        question=question,
    )
    expert = OpenAI(model, stops_at=["/n", "."])(meta_prompt)

    prompt = compose(
        """
        ${meta_prompt}${expert}.

        For instance,${expert} would answer
        """,
        meta_prompt=meta_prompt,
        expert=expert,
    )
    answer = OpenAI(model)(prompt)

    return prompt, answer


def run_example(model_fn, question, model):
    print("\n-----------------------------------------\n")
    question_s = outlines.text.string()
    fn = outlines.chain([question_s], model_fn(question_s, model))
    prompt, answer = fn(question)
    print(f"{prompt}{answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Meta Prompting examples")
    parser.add_argument(
        "--model",
        type=str,
        default="text-davinci-001",
        help="The Large Language Model to use to run the examples.",
    )
    args = parser.parse_args()

    math_q = "f(x) = x*x. What is f(f(3))?"
    sat_q = compose(
        """
    Directions: In the following question, a related
    pair of words or phrases is followed by five
    pairs of words or phrases. Choose the pair
    that best expresses a relationship similar to
    that in the original pair.
    BRAGGART :: MODESTY
    A) FLEDGLING : EXPERIENCE
    B) EMBEZZLER : GREED
    C) WALLFLOWER : TIMIDITY
    D) INVALID : MALADY
    E) CANDIDATE : AMBITION

    """
    )
    alignment_q = "What should humankind do to ensure that artificial general intelligence is aligned?"
    meaning_q = "What is the meaning of life?"

    run_example(split_into_steps, math_q, args.model)
    run_example(split_into_steps, sat_q, args.model)
    run_example(fill_in_the_blanks, sat_q, args.model)
    run_example(ask_an_expert, alignment_q, args.model)
    run_example(ask_an_expert_simple, meaning_q, args.model)
