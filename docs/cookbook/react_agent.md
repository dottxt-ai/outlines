# ReAct Agent

This example shows how to use [outlines](https://dottxt-ai.github.io/outlines/) to build your own agent with open weights local models and structured outputs. It is inspired by the blog post [A simple Python implementation of the ReAct pattern for LLMs](https://til.simonwillison.net/llms/python-react-pattern) by [Simon Willison](https://simonwillison.net/).

The ReAct pattern (for Reason+Act) is described in the paper [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). It's a pattern where you implement additional actions that an LLM can take - searching Wikipedia or running calculations for example - and then teach it how to request the execution of those actions, and then feed their results back into the LLM.

Additionally, we give the LLM the possibility of using a scratchpad described in the paper [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114) which improves the ability of LLMs to perform multi-step computations.

We use [llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) library. Outlines supports llama-cpp-python, but we need to install it ourselves:

```bash
pip install llama-cpp-python
```

We download the model weights by passing the name of the repository on the HuggingFace Hub, and the filenames (or glob pattern):
```python
import llama_cpp
from outlines import generate, models

model = models.llamacpp("NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
            "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
            tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "NousResearch/Hermes-2-Pro-Llama-3-8B"
            ),
            n_gpu_layers=-1,
            flash_attn=True,
            n_ctx=8192,
            verbose=False)
```

??? note "(Optional) Store the model weights in a custom folder"

    By default the model weights are downloaded to the hub cache but if we want so store the weights in a custom folder, we pull a quantized GGUF model [Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF) by [NousResearch](https://nousresearch.com/) from [HuggingFace](https://huggingface.co/):

    ```bash
    wget https://hf.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf
    ```

    We initialize the model:

    ```python
    import llama_cpp
    from llama_cpp import Llama
    from outlines import generate, models

    llm = Llama(
        "/path/to/model/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "NousResearch/Hermes-2-Pro-Llama-3-8B"
        ),
        n_gpu_layers=-1,
        flash_attn=True,
        n_ctx=8192,
        verbose=False
    )
    ```

## Build a ReAct agent

In this example, we use two tools:

- wikipedia: \<search term\> - search Wikipedia and returns the snippet of the first result
- calculate: \<expression\> - evaluate an expression using Python's eval() function

```python
import httpx

def wikipedia(q):
    return httpx.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query",
        "list": "search",
        "srsearch": q,
        "format": "json"
    }).json()["query"]["search"][0]["snippet"]


def calculate(numexp):
    return eval(numexp)
```

We define the logic of the agent through a Pydantic class. First, we want the LLM to decide only between the two previously defined tools:

```python
from enum import Enum

class Action(str, Enum):
    wikipedia = "wikipedia"
    calculate = "calculate"
```

Our agent will loop through Thought and Action. We explicitly give the Action Input field so it doesn't forget to add the arguments of the Action. We also add a scratchpad (optional).

```python
from pydantic import BaseModel, Field

class Reason_and_Act(BaseModel):
    Scratchpad: str = Field(..., description="Information from the Observation useful to answer the question")
    Thought: str = Field(..., description="It describes your thoughts about the question you have been asked")
    Action: Action
    Action_Input: str = Field(..., description="The arguments of the Action.")
```

Our agent will reach a Final Answer. We also add a scratchpad (optional).

```python
class Final_Answer(BaseModel):
    Scratchpad: str = Field(..., description="Information from the Observation useful to answer the question")
    Final_Answer: str = Field(..., description="Answer to the question grounded on the Observation")
```

Our agent will decide when it has reached a Final Answer and therefore to stop the loop of Thought and Action.

```python
from typing import Union

class Decision(BaseModel):
    Decision: Union[Reason_and_Act, Final_Answer]
```

We could generate a response using the json schema but we will use the regex and check that everything is working as expected:

```python
from outlines.integrations.utils import convert_json_schema_to_str
from outlines.fsm.json_schema import build_regex_from_schema

json_schema = Decision.model_json_schema()
schema_str = convert_json_schema_to_str(json_schema=json_schema)
regex_str = build_regex_from_schema(schema_str)
print(regex_str)
# '\\{[ ]?"Decision"[ ]?:[ ]?(\\{[ ]?"Scratchpad"[ ]?:[ ]?"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\])*"[ ]?,[ ]?"Thought"[ ]?:[ ]?"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\])*"[ ]?,[ ]?"Action"[ ]?:[ ]?("wikipedia"|"calculate")[ ]?,[ ]?"Action_Input"[ ]?:[ ]?"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\])*"[ ]?\\}|\\{[ ]?"Scratchpad"[ ]?:[ ]?"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\])*"[ ]?,[ ]?"Final_Answer"[ ]?:[ ]?"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\])*"[ ]?\\})[ ]?\\}'
```

We then need to adapt our prompt to the [Hermes prompt format for JSON schema](https://github.com/NousResearch/Hermes-Function-Calling?tab=readme-ov-file#prompt-format-for-json-mode--structured-outputs) and explain the agent logic:

```python
import datetime

def generate_hermes_prompt(question, schema=""):
    return (
        "<|im_start|>system\n"
        "You are a world class AI model who answers questions in JSON with correct Pydantic schema. "
        f"Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema>\n"
        "Today is " + datetime.datetime.today().strftime('%Y-%m-%d') + ".\n" +
        "You run in a loop of Scratchpad, Thought, Action, Action Input, PAUSE, Observation. "
        "At the end of the loop you output a Final Answer. "
        "Use Scratchpad to store the information from the Observation useful to answer the question "
        "Use Thought to describe your thoughts about the question you have been asked "
        "and reflect carefully about the Observation if it exists. "
        "Use Action to run one of the actions available to you. "
        "Use Action Input to input the arguments of the selected action - then return PAUSE. "
        "Observation will be the result of running those actions. "
        "Your available actions are:\n"
        "calculate:\n"
        "e.g. calulate: 4**2 / 3\n"
        "Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary\n"
        "wikipedia:\n"
        "e.g. wikipedia: Django\n"
        "Returns a summary from searching Wikipedia\n"
        "DO NOT TRY TO GUESS THE ANSWER. Begin! <|im_end|>"
        "\n<|im_start|>user\n" + question + "<|im_end|>"
        "\n<|im_start|>assistant\n"
    )
```

We define a ChatBot class

```python
class ChatBot:
    def __init__(self, prompt=""):
        self.prompt = prompt

    def __call__(self, user_prompt):
        self.prompt += user_prompt
        result = self.execute()
        return result

    def execute(self):
        generator = generate.regex(model, regex_str)
        result = generator(self.prompt, max_tokens=1024, temperature=0, seed=42)
        return result
```

We define a query function:

```python
import json

def query(question, max_turns=5):
    i = 0
    next_prompt = (
        "\n<|im_start|>user\n" + question + "<|im_end|>"
        "\n<|im_start|>assistant\n"
    )
    previous_actions = []
    while i < max_turns:
        i += 1
        prompt = generate_hermes_prompt(question=question, schema=Decision.model_json_schema())
        bot = ChatBot(prompt=prompt)
        result = bot(next_prompt)
        json_result = json.loads(result)['Decision']
        if "Final_Answer" not in list(json_result.keys()):
            scratchpad = json_result['Scratchpad'] if i == 0 else ""
            thought = json_result['Thought']
            action = json_result['Action']
            action_input = json_result['Action_Input']
            print(f"\x1b[34m Scratchpad: {scratchpad} \x1b[0m")
            print(f"\x1b[34m Thought: {thought} \x1b[0m")
            print(f"\x1b[36m  -- running {action}: {str(action_input)}\x1b[0m")
            if action + ": " + str(action_input) in previous_actions:
                observation = "You already run that action. **TRY A DIFFERENT ACTION INPUT.**"
            else:
                if action=="calculate":
                    try:
                        observation = eval(str(action_input))
                    except Exception as e:
                        observation = f"{e}"
                elif action=="wikipedia":
                    try:
                        observation = wikipedia(str(action_input))
                    except Exception as e:
                        observation = f"{e}"
            print()
            print(f"\x1b[33m Observation: {observation} \x1b[0m")
            print()
            previous_actions.append(action + ": " + str(action_input))
            next_prompt += (
                "\nScratchpad: " + scratchpad +
                "\nThought: " + thought +
                "\nAction: " + action  +
                "\nAction Input: " + action_input +
                "\nObservation: " + str(observation)
            )
        else:
            scratchpad = json_result["Scratchpad"]
            final_answer = json_result["Final_Answer"]
            print(f"\x1b[34m Scratchpad: {scratchpad} \x1b[0m")
            print(f"\x1b[34m Final Answer: {final_answer} \x1b[0m")
            return final_answer
    print(f"\nFinal Answer: I am sorry, but I am unable to answer your question. Please provide more information or a different question.")
    return "No answer found"
```

We can now test our ReAct agent:

```python
print(query("What's 2 to the power of 10?"))
# Scratchpad:
# Thought: I need to perform a mathematical calculation to find the result of 2 to the power of 10.
#  -- running calculate: 2**10
#
# Observation: 1024
#
# Scratchpad: 2 to the power of 10 is 1024.
# Final Answer: 2 to the power of 10 is 1024.
# 2 to the power of 10 is 1024.
```

```python
print(query("What does England share borders with?"))
# Scratchpad:
# Thought: To answer this question, I will use the 'wikipedia' action to gather information about England's geographical location and its borders.
#  -- running wikipedia: England borders
#
# Observation: Anglo-Scottish <span class="searchmatch">border</span> (Scottish Gaelic: Crìochan Anglo-Albannach) is an internal <span class="searchmatch">border</span> of the United Kingdom separating Scotland and <span class="searchmatch">England</span> which runs for
#
# Scratchpad: Anglo-Scottish border (Scottish Gaelic: Crìochan Anglo-Albannach) is an internal border of the United Kingdom separating Scotland and England which runs for
# Final Answer: England shares a border with Scotland.
# England shares a border with Scotland.
```

As mentioned in Simon's blog post, this is not a very robust implementation at all and there's a ton of room for improvement. But it is lovely how simple it is with a few lines of Python to make these extra capabilities available to the LLM. And now you can run it locally with an open weights LLM.

This example was originally contributed by [Alonso Silva](https://github.com/alonsosilvaallende).
