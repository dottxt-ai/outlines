# Generate structured output for audio understanding

Even though audio-LM models for audio-text-to-text tasks are still pretty niche, they are still useful (and fun) to analyse, extract informations, translate or transcript speeches.

This cookbook highlights the new integration of audio-LM and has been tested with `Qwen/Qwen2-Audio-7B-Instruct` ([HF link](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)).

## Setup

As usual let's have the right packages

```bash
pip install outlines torch==2.4.0 transformers accelerate librosa
```

So that you can import as follow:

```python
# LLM stuff
import outlines
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# Audio stuff
import librosa
from io import BytesIO
from urllib.request import urlopen

# Some ooo stuff
from enum import Enum
from pydantic import BaseModel
from typing import Optional
```

## Load the model and processor

To achieve audio analysis we will need a model and its processor to pre-process prompts and audio. Let's do as follow:

```python
qwen2_audio = outlines.models.transformers_vision(
    "Qwen/Qwen2-Audio-7B-Instruct",
    model_class=Qwen2AudioForConditionalGeneration,
    model_kwargs={
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    },
    processor_kwargs={
        "device": "cuda", # set to "cpu" if you don't have a GPU
    },
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
```

Let's also define a useful audio extractor from conversational prompts:

```pyton
def audio_extractor(conversation):
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for elt in message["content"]:
                if elt["type"] == "audio":
                    audios.append(
                        librosa.load(
                            BytesIO(urlopen(elt['audio_url']).read()),
                            sr=processor.feature_extractor.sampling_rate
                        )[0]
                    )
    return audios
```

## Question answering

Let's say we want to analyse and answer the question of the lady in this [audio](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav).

### Data structure

To have a structured data output, we can define the following data model:

```python
class Age(int, Enum):
    twenties = 20
    fifties = 50

class Gender(str, Enum):
    male = "male"
    female = "female"

class Person(BaseModel):
    gender: Gender
    age: Age
    language: Optional[str]
```

### Prompting

Let's have the following prompt to ask our model:

```python
audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": audio_url},
        {
            "type": "text",
            "text": f"""As asked in the audio, what is the gender and the age of the speaker?

            Return the information in the following JSON schema:
            {Person.model_json_schema()}
            """
        },
    ]},
]
```

But we cannot pass it raw! We need to pre-process it and handle the audio file.

```python
audios = audio_extractor(conversation)

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
```

Now we're ready to ask our model!

### Run the model

As usual with the outlines' framework, we will instantiate a generator that specifically struture the output based on our data model:

```python
person_generator = outlines.generate.json(
    qwen2_audio,
    Person,
    sampler=outlines.samplers.greedy()
)
```

That runs just like:

```python
result = person_generator(prompt, audios)
```

And you are expecting to get a result as follow:
```
Person(
    gender=<Gender.female: 'female'>,
    age=<Age.twenties: 20>,
    language='English'
)
```

## Classification

Now we can focus on this [audio](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3) of a glass breaking.

The integration of audio transformers, allows you to use all the functionalities of the outlines' API such as the `choice` method. We can do as follow:

### Prompting

Let's consider the following prompt and pre-process our audio:

```python
audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": audio_url},
        {
            "type": "text",
            "text": "Do you hear a dog barking or a glass breaking?"
        },
    ]},
]

audios = audio_extractor(conversation)

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
```

### Run the model

As mentioned, we will use the `choice` method to generate our structured output:

```python
choice_generator = outlines.generate.choice(
    qwen2_audio,
    ["dog barking", "glass breaking"],
)

result = choice_generator(prompt, audios)
```

And you are expected to have:
```python
print(result)
# "glass breaking"
```
