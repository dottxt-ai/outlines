from dataclasses import dataclass
from enum import Enum

import torch
from torch import Generator
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
    job: str
    # Ignore mypy checks here because it still doesn't support conlist or constr: https://github.com/pydantic/pydantic/issues/975
    interests: conlist(str, min_length=1)  # type: ignore


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
            job="Famous Soccer Player -> Famous Author",
            interests=["Soccer", "Travel", "Friends", "Books", "Fluffy Animals"],
        ),
    ),
    Example(
        description="I run my company and build houses for a living. I'm a big fan of the outdoors and love to go hiking, camping, and fishing. I don't like video games, but do like to watch movies. My love language is home-cooked food, and I'm looking for someone who isn't afraid to get their hands dirty.",
        profile=DatingProfile(
            job="House Construction Manager / Entrepreneur",
            interests=["Hunting", "Hiking", "The outdoors", "Home-cooked food"],
        ),
    ),
    Example(
        description="I run my own Youtube channel with 10M subscribers. I love working with kids, and my audience skews pretty young too. In my free time, I play Fortnite and Roblox. I'm looking for someone who is also a gamer and likes to have fun. I'm learning Japanese in my free time as well as how to cook.",
        profile=DatingProfile(
            job="Youtuber 10M+ subscribers",
            interests=["Kids", "Gaming", "Japanese"],
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
rng = Generator(device='cuda')
rng = rng.manual_seed(21712121)
prompt = dating_profile_prompt(description=new_description, examples=samples)
from outlines.generate.samplers import greedy
profile = outlines.generate.json(model, DatingProfile, sampler=greedy)(prompt, rng=rng)  # type: ignore
print(profile)
