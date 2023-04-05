"""Meta-prompting examples.

References
----------

.. [0] "Prompting is programming: A Query Language for Large Language Models"
       https://arxiv.org/abs/2212.06094
.. [1] "Prompt programming For Large Language Models: Beyond the Few-Shot Paradigm"
       https://arxiv.org/abs/2102.07350.

"""
import outlines
from outlines import compose
from outlines.text.models.openai import OpenAI


def split_into_steps(question, llm):
    prompt = compose(
        """
        ${question}
        Let's solve this problem by splitting it into steps.
        """,
        question=question,
    )
    return llm(prompt)


def fill_in_the_blanks(question, llm):
    meta_prompt = compose(
        """
        ${question}
        To solve this problem, we will analyze each of the options and determine
        """,
        question=question,
    )
    goal = llm(meta_prompt)

    prompt = compose(
        """
        ${meta_prompt}${goal}. Let's begin.
        """,
        meta_prompt=meta_prompt,
        goal=goal,
    )
    answer = llm(prompt)

    return goal, answer


def ask_an_expert(question, llm):
    prompt = compose(
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
    expert = llm(prompt, stops_at=['""'])

    prompt = compose(
        """
        ${prompt}${expert}
        I am ready to ask my question.
        "${expert} I say,
        ${question}
        """,
        prompt=prompt,
        expert=expert,
        question=question,
    )
    answer = llm(prompt)
    return prompt, expert, answer


def ask_an_expert_simple(question, llm):
    meta_prompt = compose(
        """
        Q: ${question}
        A: A good person to answer this question would be
        """,
        question=question,
    )
    expert = llm(meta_prompt, stops_at=["/n", "."])

    prompt = compose(
        """
        ${meta_prompt}${expert}

        For instance,${expert} would answer
        """,
        meta_prompt=meta_prompt,
        expert=expert,
    )
    answer = llm(prompt)

    return answer


llm = OpenAI("text-davinci-001")
fn = outlines.chain([], ask_an_expert_simple("What is the meaning of life?", llm))
fn = outlines.chain([], split_into_steps("f(x) = x*x. What is f(f(3))?", llm))
fn = outlines.chain(
    [],
    ask_an_expert(
        "What should humankind do to ensure that artificial general intelligence is aligned?",
        llm,
    ),
)

direction = compose(
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
fn = outlines.chain([], split_into_steps(direction, llm))
fn = outlines.chain([], fill_in_the_blanks(direction, llm))
