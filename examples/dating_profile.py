from dataclasses import dataclass
from enum import Enum

import torch
import transformers
from pydantic import BaseModel, conlist

import outlines
from outlines import models


class QuestionChoice(str, Enum):
    A = "The key to my heart is"
    B = "The first item on my bucket list is"
    C = "Perks of dating me"
    D = "Message me if you also love"
    E = "People would describe me as"
    F = "I can beat you in a game of"


@dataclass
class QuestionAnswer:
    question: QuestionChoice
    answer: str


class DatingProfile(BaseModel):
    # It is possible put length constraints on these strings using constr- however, this appears to dramatically increase the generation time
    # This may be resolved in the future with this PR: https://github.com/outlines-dev/outlines/pull/272
    bio: str
    job: str
    # Ignore mypy checks here because it still doesn't support conlist or constr: https://github.com/pydantic/pydantic/issues/975
    interests: conlist(str, min_length=1, max_length=5)  # type: ignore
    qna1: QuestionAnswer
    qna2: QuestionAnswer


@dataclass
class Example:
    description: str
    profile: DatingProfile


@outlines.prompt
def dating_profile_prompt(description: str, examples: list[Example]):
    """
    You are a world-renowned matchmaker who understands the modern dating market. Your job is to generate dating app profiles for male clients interested in women based on a provided description. The profiles should be authentic, show off their strengths, and maximize their likelihood of getting matches on dating apps.
    Here are some examples of past clients that you have successfully created profiles for:
    {% for example in examples %}
    Description:
    {{ example.description }}
    Profile:
    {{ example.profile }}
    {% endfor %}
    Here is the new client who you need to create a profile for:
    Description: {{ description }}
    Profile:
    """


samples: list[Example] = [
    Example(
        description="I'm an author and former professional soccer player living in Seattle who publishes popular fiction books. A typical day for me starts by hanging out with my cat, drinking a coffee, and reading as much as I can in a few hours. Then, I'll prepare a quick smoothie before starting to write for a few hours, take a break with soccer or running a few miles, and finally meet friends for dinner at a new, hip restaurant in the evening. Sometimes we go axe-throwing afterwards, or play poker, or watch a comedy show, or visit a dive bar. On my vacations, I travel extensively to countries South America, Europe, and Asia, with the goal of visiting them all!",
        profile=DatingProfile(
            bio="Adventurer, dreamer, author, and soccer enthusiast. Life’s too short to waste time so I make the most of each day by exploring new places and playing with my friends on the pitch. What’s your favorite way to get out and have fun?",
            job="Famous Soccer Player -> Famous Author",
            interests=["Soccer", "Travel", "Friends", "Books", "Fluffy Animals"],
            qna1=QuestionAnswer(
                question=QuestionChoice.B, answer="swim in all seven oceans!"
            ),
            qna2=QuestionAnswer(
                question=QuestionChoice.E,
                answer="fun-loving, adventurous, and a little bit crazy",
            ),
        ),
    ),
    Example(
        description="I run my company and build houses for a living. I'm a big fan of the outdoors and love to go hiking, camping, and fishing. I don't like video games, but do like to watch movies. My love language is home-cooked food, and I'm looking for someone who isn't afraid to get their hands dirty.",
        profile=DatingProfile(
            bio="If you're looking for a Montana man who loves to get outdoors and hunt, and who's in-tune with his masculinity then I'm your guy!",
            job="House Construction Manager / Entrepreneur",
            interests=["Hunting", "Hiking", "The outdoors", "Home-cooked food"],
            qna1=QuestionAnswer(question=QuestionChoice.A, answer="food made at home"),
            qna2=QuestionAnswer(
                question=QuestionChoice.C,
                answer="having a man in your life who can fix anything",
            ),
        ),
    ),
    Example(
        description="I run my own Youtube channel with 10M subscribers. I love working with kids, and my audience skews pretty young too. In my free time, I play Fortnite and Roblox. I'm looking for someone who is also a gamer and likes to have fun. I'm learning Japanese in my free time as well as how to cook.",
        profile=DatingProfile(
            bio="Easy on the eyes (find me on Youtube!) and great with kids. What more do you need?",
            job="Youtuber 10M+ subscribers",
            interests=["Kids", "Gaming", "Japanese"],
            qna1=QuestionAnswer(question=QuestionChoice.D, answer="anime and gaming!"),
            qna2=QuestionAnswer(question=QuestionChoice.F, answer="Fortnite, gg ez"),
        ),
    ),
]


# Below requires ~13GB of GPU memory
# https://huggingface.co/mosaicml/mpt-7b-8k-instruct
# Motivation: Reasonably large model that fits on a single GPU and has been fine-tuned for a larger context window
config = transformers.AutoConfig.from_pretrained(
    "mosaicml/mpt-7b-8k-instruct", trust_remote_code=True
)
config.init_device = "meta"
model = models.transformers(
    model_name="mosaicml/mpt-7b-8k-instruct",
    device="cuda",
    model_kwargs={
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
    },
)

new_description = "I'm a laid-back lawyer who spends a lot of his free-time gaming. I work in a corporate office, but ended up here after the start-up I cofounded got acquired, so still play ping pong with my cool coworkers every day. I have a bar at home where I make cocktails, which is great for entertaining friends. I secretly like to wear suits and get a new one tailored every few months. I also like weddings because I get to wear those suits, and it's a good excuse for a date. I watch the latest series because I'm paying, with my hard-earned money, for every streaming service."

prompt = dating_profile_prompt(description=new_description, examples=samples)
profile = outlines.generate.json(model, DatingProfile)(prompt)  # type: ignore
print(profile)

# Sample generated profiles
"""
{
    "bio": "I'm an ambitious lawyer with a casual and fashionable style. I love games and sports, but my true passion is preparing refreshing cocktails at home and dressing to the nines at weddings. I'm currently looking for a woman to show a good time to and get a kiss on the opulent suit I just had made. Send resumÃ € to this inbox.",
    "job": "Lawyer",
    "interests":
    [
        "Stylish guys",
        "Gaming",
        "Ping pong",
        "Cocktails",
        "Weddings"
    ],
    "qna1":
    {
        "question": "The first item on my bucket list is",
        "answer": "be married and have a family."
    },
    "qna2":
    {
        "question": "People would describe me as",
        "answer": "charming, stylish, and funny."
    }
}
"""

"""
{
    "bio": "I’m a sexy lawyer with time on my hands. I love to game and play ping pong, but the real reason you should swipe to the right is because I look great in a suit. Who doesn’t love a man in a suit? Just saying. Send me a message if you think it’s time to take your dating life to the next level.",
    "job": "Lawyer",
    "interests":
    [
        "Gaming",
        "Ping Pong",
        "Tailored Suits",
        "Weddings",
        "Streaming Services"
    ],
    "qna1":
    {
        "question": "The first item on my bucket list is",
        "answer": "simulate space but stay alive for as long as possible"
    },
    "qna2":
    {
        "question": "People would describe me as",
        "answer": "easy-going, a little nerdy but with a mature essence"
    }
}
"""
