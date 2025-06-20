# Generate a synthetic dating profile from a description

In this example we will see how we can use Outlines to generate synthetic data for a dating application. This example was originally contributed by [Vibhor Kumar](https://github.com/veezbo).

```python
import json
from dataclasses import dataclass
from enum import Enum

import torch
import transformers
from pydantic import BaseModel, conlist, constr

import outlines
```

## Defining the profile with Pydantic

Here a dating profile will consist in a biography, a job, a list of interests and two question-answer pairs. The questions are written in advance by the team, and the users are asked to provide an answer:

```python
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
```

Users need to provide a short biography, with a minimum of 10 and a maximum of 300 characters. The application also limits job descriptions to 50 characters. In addition to the question-answer pairs, the user is required to provide a list of between 1 and 5 interests:

```python
class DatingProfile(BaseModel):
    bio: constr(str, min_length=10, max_length=300)
    job: constr(str, max_lengt=50)
    interests: conlist(str, min_length=1, max_length=5)  # type: ignore
    qna1: QuestionAnswer
    qna2: QuestionAnswer
```

## Prompt template and examples

We will ask the model to generate profiles from a high-level description:

```python
@dataclass
class Example:
    description: str
    profile: DatingProfile
```

We will use Outlines' prompt templating abilities to generate the prompt for us. This help clearly separate the general prompting logic from what is specific to an example.

```python
from outlines import Template

dating_profile_prompt = Template.from_string(
    """
    You are a world-renowned matchmaker who understands the modern dating
    market. Your job is to generate dating app profiles for male clients
    interested in women based on a provided description. The profiles should be
    authentic, show off their strengths, and maximize their likelihood of
    getting matches on dating apps.  Here are some examples of past clients that
    you have successfully created profiles for:

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
)
```

We will provide the model with several few-shot examples:

```python
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
```

## Load the model

We will use Mosaic's MPT-7B model (requires 13GB of GPU memory) which can fit on a single GPU with a reasonable context window. We initialize it with Outlines:

```python
MODEL_NAME = "mosaicml/mpt-7b-8k-instruct"

config = transformers.AutoConfig.from_pretrained(
    MODEL_NAME, trust_remote_code=True
)
config.init_device = "meta"
model_kwargs = {
    "config": config,
    "trust_remote_code": True,
    "torch_dtype": torch.bfloat16,
    "device_map": "cuda",
}
tf_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
tf_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = outlines.from_transformers(tf_model, tokenizer=tf_tokenizer)
```

## JSON-structured generation of profiles

We will now generate a dating profile from a textual description of oneself:

``` python
new_description = """I'm a laid-back lawyer who spends a lot of his free-time
gaming. I work in a corporate office, but ended up here after the start-up  I
cofounded got acquired, so still play ping pong with my cool coworkers every
day.  I have a bar at home where I make cocktails, which is great for
entertaining  friends. I secretly like to wear suits and get a new one tailored
every few  months. I also like weddings because I get to wear those suits, and
it's  a good excuse for a date. I watch the latest series because I'm paying,
with my hard-earned money, for every streaming service."""

prompt = dating_profile_prompt(description=new_description, examples=samples)
profile = model(prompt, DatingProfile)
parsed_profile = DatingProfile.model_validate_json(json.loads(profile))
```

## Results

Here are a couple of results:

```json
{
    "bio": """I'm an ambitious lawyer with a casual and fashionable style. I love
    games and sports, but my true passion is preparing refreshing cocktails at
    home and dressing to the nines at weddings. I'm currently looking for a woman
    to show a good time to and get a kiss on the opulent suit I just had made.
    Send resume to this inbox.""",
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
```

```json
{
    "bio": """I’m a sexy lawyer with time on my hands. I love to game and
    play ping pong, but the real reason you should swipe to the right
    is because I look great in a suit. Who doesn’t love a man in a
    suit? Just saying. Send me a message if you think it’s time to take
    your dating life to the next level.""",
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
```
