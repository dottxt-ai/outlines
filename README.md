<div align="center" style="margin-bottom: 1em;">

<img src="./docs/assets/images/logo.png" alt="Outlines Logo" width=300></img>


 🗒️ *Structured outputs for LLMs* 🗒️

Made with ❤👷️ by the team at [.txt](https://dottxt.co).

[![Documentation][documentation-badge]][documentation]
[![Contributors][contributors-badge]][contributors]
[![Downloads][downloads-badge]][pypistats]
[![Discord][discord-badge]][discord]

[Quickstart](#quick-start)  [Documentation](#documentation) | [Youtube channel][youtube-dottxt] | [.txt blog][blog-dottxt] | [Twitter][dottxt-twitter]


</div>

## Why Outlines?

LLMs are powerful — but their outputs are unpredictable.

Most tools try to fix bad outputs *after* generation with parsing, regex, or fragile code.

Outlines flips the approach.

### It guarantees structured outputs *during* generation — directly from any LLM.

- No token overhead
- No provider lock-in
- No fragile parsing

Just clean, reliable data — ready to use.

``` bash
pip install outlines  # Python 3.8+
```

## The Outlines Philosophy

Outlines follows a simple pattern that mirrors Python's own type system:

``` python

result = model(prompt, output_type)
```

Just pass the type you want, and Outlines ensures you get data in that structure:

- Need a yes/no answer? Pass `Literal["Yes", "No"]`
- Need an integer? Pass `int`
- Need a complex object? Pass a Pydantic model

Instead of parsing text after generation, Outlines builds the structure directly into the generation process, guaranteeing valid outputs every time.

## Quickstart

``` python
import outlines
from typing import Literal

model = outlines.from_openai("gpt-4")

result = model("Is this email positive?", Literal["Yes", "No"])
print(result)
```

## Real-world examples

Here are production-ready examples showing how Outlines solves common problems:

<details><summary><b>Customer Support Triage</b></summary>

``` python
import outlines
from enum import Enum
from pydantic import BaseModel
from typing import List

class TicketPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"

class ServiceTicket(BaseModel):
    priority: TicketPriority
    category: str
    requires_manager: bool
    summary: str
    action_items: List[str]

model = outlines.from_openai("gpt-4")

customer_email = """
Subject: URGENT - Cannot access my account after payment
 
I paid for the premium plan 3 hours ago and still can't access any features.
I've tried logging out and back in multiple times. This is unacceptable as I
have a client presentation in an hour and need the analytics dashboard.
Please fix this immediately or refund my payment.
"""

ticket = model(f"Analyze this customer email:\n\n{customer_email}", ServiceTicket)

# Use structured data to route the ticket
if ticket.priority == "urgent" or ticket.requires_manager:
    alert_manager(ticket)
```
</details>

<details><summary><b>E-commerce product categorization</b></summary>

``` python
import outlines
from pydantic import BaseModel
from typing import List, Optional

class ProductCategory(BaseModel):
    main_category: str
    sub_category: str
    attributes: List[str]
    brand_match: Optional[str]

model = outlines.from_vllm("http://localhost:8000")  # Running Mixtral

# Process product descriptions in batches
product_descriptions = [
    "Apple iPhone 15 Pro Max 256GB Titanium, 6.7-inch Super Retina XDR display with ProMotion",
    "Organic Cotton T-Shirt, Men's Medium, Navy Blue, 100% Sustainable Materials",
    "KitchenAid Stand Mixer, 5 Quart, Red, 10-Speed Settings with Dough Hook Attachment"
]

# Get structured categorization for all products
categories = model.generate_batch(
    [f"Categorize this product: {desc}" for desc in product_descriptions],
    ProductCategory
)

# Use categorization for inventory management
for product, category in zip(product_descriptions, categories):
    update_inventory(product, category.main_category, category.sub_category)
```
</details>

<details><b><summary>Extracting data from medical notes</b></summary>
```python
import outlines
import pandas as pd
import json
from typing import List, Dict

# Sample unstructured medical notes
medical_notes = [
    """
    Patient consulted on 04/15/2023. 37yo male presenting with persistent cough for 2 weeks. 
    Temp: 99.8F, BP: 128/82, HR: 76. Patient reports taking Lisinopril 10mg daily for hypertension. 
    Lungs clear on auscultation. No signs of pneumonia. Diagnosis: Acute bronchitis. 
    Plan: Prescribed Benzonatate 200mg TID PRN for cough for 7 days. F/U in 1 week if symptoms persist.
    """,
    
    """
    Follow-up visit on 5-22-23. Female patient, 42 years old with history of migraines. Reports headaches 
    have decreased from 4x/week to 1x/week since starting Topamax 50 mg BID. BP 118/74, pulse 68. 
    Patient also takes Vitamin D 2000 IU daily and Magnesium 400mg QD. Assessment: Improving migraine disorder. 
    Plan: Continue current regimen. Return in 3 months for evaluation.
    """
]

# Define regex patterns for clinical data
VITAL_PATTERN = r"(?:BP|Blood Pressure):\s*(\d{2,3}/\d{2,3})"
MEDICATION_PATTERN = r"([A-Z][a-z]+(?:in|ol|ide|ate|ine|one|zole|pril|sartan|mab|nib|prazole|dipine|statin)[a-z]*)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|IU|mEq)"
DATE_PATTERN = r"(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:\d{1,2}(?:st|nd|rd|th)? [A-Z][a-z]{2,8},? \d{4})"
DIAGNOSIS_PATTERN = r"(?:Diagnosis|Assessment):(.+?)(?:\.|Plan|\n)"

model = outlines.from_ollama("mistral")  # Or any other model

def extract_structured_medical_data(notes: List[str]) -> List[Dict]:
    """Extract structured medical data using LLM with regex constraints"""
    all_records = []
    
    for note in notes:
        record = {}
        
        # Extract date with context understanding
        prompt = f"What is the date of the patient visit in this note? Extract only the date.\n\n{note}"
        record["visit_date"] = model(prompt, outlines.Regex(DATE_PATTERN))
        
        # Extract vitals - LLM helps identify which values to extract when multiple exist
        prompt = f"What is the patient's blood pressure reading in this note? Format as XXX/XX.\n\n{note}"
        record["blood_pressure"] = model(prompt, outlines.Regex(VITAL_PATTERN))
        record["blood_pressure"] = record["blood_pressure"].replace("BP:", "").replace("Blood Pressure:", "").strip()
        
        # Extract age and gender - LLM understands different formats
        prompt = f"Extract the patient's age and gender from this note. Format as 'XX-year-old gender'.\n\n{note}"
        record["patient"] = model(prompt, outlines.Regex(r"\d{1,3}(?:-|\s)?(?:year|y)(?:-|\s)?old\s(?:male|female|non-binary|transgender)"))
        
        # Extract medications - LLM understands context to identify actual medications
        prompt = f"""
        Extract all medications mentioned in this note, one at a time.
        Include only name, dosage, and unit (like 'Lisinopril 10mg'). 
        
        Note:
        {note}
        
        First medication:
        """
        medications = []
        
        # Multiple medications may exist, so we extract sequentially
        for i in range(5):  # Assume maximum 5 medications
            try:
                medication = model(prompt, outlines.Regex(MEDICATION_PATTERN), max_tokens=20)
                medications.append(medication)
                prompt = "Next medication (if any):"
            except Exception:
                # No more medications found
                break
                
        record["medications"] = medications
        
        # Extract diagnosis - LLM can understand the diagnosis even with complex formatting
        prompt = f"What is the diagnosis or assessment for this patient?\n\n{note}"
        diagnosis = model(prompt, outlines.Regex(DIAGNOSIS_PATTERN))
        record["diagnosis"] = diagnosis.replace("Diagnosis:", "").replace("Assessment:", "").strip()
        
        # Use LLM to summarize the plan without regex - showing hybrid approach
        prompt = f"What is the treatment plan for this patient? Summarize briefly.\n\n{note}"
        record["plan"] = model(prompt, max_tokens=50)
        
        all_records.append(record)
    
    return all_records

# Process all notes
structured_data = extract_structured_medical_data(medical_notes)

# Convert to DataFrame for analysis
df = pd.DataFrame(structured_data)

# Output as JSON
print(json.dumps(structured_data, indent=2))

# Use structured data for downstream tasks
def analyze_medical_records(records):
    """Example of using the extracted structured data"""
    # Count patients by diagnosis
    diagnosis_counts = {}
    for record in records:
        diagnosis = record["diagnosis"]
        diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
    
    # Find most common medications
    all_meds = []
    for record in records:
        all_meds.extend(record["medications"])
    
    med_counts = {}
    for med in all_meds:
        med_name = med.split()[0]  # Extract just the name
        med_counts[med_name] = med_counts.get(med_name, 0) + 1
    
    return {
        "diagnosis_summary": diagnosis_counts,
        "medication_summary": med_counts
    }

# Run analysis on structured data
analysis = analyze_medical_records(structured_data)
print("\nAnalysis Results:")
print(json.dumps(analysis, indent=2))
```
</details>
<details>
<summary><b>Robust Information Extraction with Union Types</b></summary>

```python
import outlines
from typing import Union, List, Literal
from pydantic import BaseModel
from enum import Enum


class EventType(str, Enum):
    conference = "conference"
    webinar = "webinar"
    workshop = "workshop"
    meetup = "meetup"
    other = "other"


class EventInfo(BaseModel):
    """Structured information about a tech event"""
    name: str
    date: str
    location: str
    event_type: EventType
    topics: List[str]
    registration_required: bool


# Create a union type that can either be a structured EventInfo or "I don't know"
EventResponse = Union[EventInfo, Literal["I don't know"]]

# Sample event descriptions
event_descriptions = [
    # Complete information
    """
    Join us for DevCon 2023, the premier developer conference happening on November 15-17, 2023 
    at the San Francisco Convention Center. Topics include AI/ML, cloud infrastructure, and web3. 
    Registration is required.
    """,
    
    # Insufficient information
    """
    Tech event next week. More details coming soon!
    """
]

# Initialize model
model = outlines.from_openai("gpt-3.5-turbo")

# Process events
results = []
for description in event_descriptions:
    prompt = f"Extract structured information about this tech event:\n\n{description}"
    # Union type allows the model to return structured data or "I don't know"
    result = model(prompt, EventResponse)
    results.append(result)

# Display results
for i, result in enumerate(results):
    print(f"Event {i+1}:")
    if isinstance(result, str):
        print(f"  {result}")
    else:
        # It's an EventInfo object
        print(f"  Name: {result.name}")
        print(f"  Type: {result.event_type}")
        print(f"  Date: {result.date}")
        print(f"  Topics: {', '.join(result.topics)}")
    print()

# Use structured data in downstream processing
structured_count = sum(1 for r in results if isinstance(r, EventInfo))
print(f"Successfully extracted data for {structured_count} of {len(results)} events")
```
</details>

<details>
<summary><b>Simple Document Classification with Literal</b></summary>

```python
import outlines
from typing import Literal, List
import pandas as pd

# Define classification categories using Literal
DocumentCategory = Literal[
    "Financial Report", 
    "Legal Contract", 
    "Technical Documentation",
    "Marketing Material",
    "Personal Correspondence"
]

# Sample documents to classify
documents = [
    "Q3 Financial Summary: Revenue increased by 15% year-over-year to $12.4M. EBITDA margin improved to 23% compared to 19% in Q3 last year. Operating expenses...",
    
    "This agreement is made between Party A and Party B, hereinafter referred to as 'the Parties', on this day of...",
    
    "The API accepts POST requests with JSON payloads. Required parameters include 'user_id' and 'transaction_type'. The endpoint returns a 200 status code on success."
]

# Initialize model
model = outlines.from_ollama("mistral")  # Or any model of your choice

# Classify documents
def classify_documents(texts: List[str]) -> List[DocumentCategory]:
    results = []
    
    for text in texts:
        prompt = f"Classify this document into exactly one category:\n\n{text[:300]}..."
        # The model must return one of the predefined categories
        category = model(prompt, DocumentCategory)
        results.append(category)
    
    return results

# Perform classification
classifications = classify_documents(documents)

# Create a simple results table
results_df = pd.DataFrame({
    "Document": [doc[:50] + "..." for doc in documents],
    "Classification": classifications
})

print(results_df)

# Count documents by category
category_counts = pd.Series(classifications).value_counts()
print("\nCategory Distribution:")
print(category_counts)
```
</details>

## Core Features

| Feature | Description | Documentation |
|---------|-------------|---------------|
| **Universal Model Support** | Works with OpenAI, Anthropic, Ollama, vLLM, Transformers, llama.cpp | [Model Integrations →](https://dottxt-ai.github.io/outlines/latest/installation) |
| **Multiple Choices** | Constrain outputs to predefined options | [Multiple Choices →](https://dottxt-ai.github.io/outlines/latest/guides/choice_generation) |
| **Type Constraints** | Force outputs to be integers, floats, etc. | [Type Constraints →](https://dottxt-ai.github.io/outlines/latest/guides/format_generation) |
| **Regex Generation** | Generate text following a regex pattern | [Regex Guide →](https://dottxt-ai.github.io/outlines/latest/guides/regex_generation) |
| **JSON/Pydantic** | Generate outputs matching JSON schemas | [JSON Guide →](https://dottxt-ai.github.io/outlines/latest/guides/json_generation) |
| **Grammars** | Enforce complex output structures | [Grammar Guide →](https://dottxt-ai.github.io/outlines/latest/guides/cfg_generation) |
| **️Function Calls** | Infer structure from function signatures | [Function Guide →](https://dottxt-ai.github.io/outlines/latest/guides/function_generation) |

## Who is using Outlines?


## Why should I use structured generation?

* It doesn't add any overhead during inference (cost-free)
* It allows Open Source models to beat closed source models ([Mistral](https://x.com/dottxtai/status/1797692104023363765), [GPT-4](https://x.com/dottxtai/status/1798443290913853770))
* [It speeds up inference](http://blog.dottxt.co/coalescence.html)
* [It improves the performance of base models (GSM8K)](http://blog.dottxt.co/performance-gsm8k.html)
* [It improves the performance of finetuned models (CoNNL)](https://predibase.com/blog/lorax-outlines-better-json-extraction-with-structured-generation-and-lora)
* [It improves model efficiency (less examples needed)](https://huggingface.co/blog/evaluation-structured-outputs)

## .txt company

<div align="center">
<img src="./docs/assets/images/dottxt.png" alt="Outlines Logo" width=100></img>
</div>

We started a company to keep pushing the boundaries of structured generation. Learn more about [.txt](https://twitter.com/dottxtai), and  [give our .json API a try](https://h1xbpbfsf0w.typeform.com/to/ZgBCvJHF) if you need a hosted solution ✨

### Chat template tokens

Outlines does not manage chat templating tokens when using instruct models. You must apply the chat template tokens to the prompt yourself. Chat template tokens are not needed for base models.

Please see [the documentation](https://dottxt-ai.github.io/outlines/latest/reference/chat_templating) on chat templating for more.

### Multiple choices

You can reduce the completion to a choice between multiple possibilities:

``` python
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

# You must apply the chat template tokens to the prompt!
# See below for an example.
prompt = """
<|im_start|>system
You extract information from text.
<|im_end|>

<|im_start|>user
What food does the following text describe?

Text: I really really really want pizza.
<|im_end|>
<|im_start|>assistant
"""

answer = model(prompt, Literal["Pizza", "Pasta", "Salad", "Dessert"])
# Likely answer: Pizza
```

You can also pass in choices with an `Enum`:

````python
from enum import Enum

class Food(str, Enum):
    pizza = "Pizza"
    pasta = "Pasta"
    salad = "Salad"
    dessert = "Dessert"

answer = model(prompt, Food)
# Likely answer: Pizza
````

### Type constraints

You can instruct the model to only return integers or floats:


``` python
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines

model_name = "WizardLM/WizardMath-7B-V1.1"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

prompt = "<s>result of 9 + 9 = 18</s><s>result of 1 + 2 = "
answer = outlines.generate.format(model, int)(prompt)
print(answer)
# 3

prompt = "sqrt(2)="
answer = model(prompt, outlines.types.number, max_tokens=10)
print(answer)
# 1.41421356
```

### Efficient regex-structured generation

Outlines also comes with fast regex-structured generation. In fact, the `choice` and
`format` functions above all use regex-structured generation under the
hood:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines
from outlines import regex

model_name = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

prompt = """
<|im_start|>system You are a helpful assistant.
<|im_end|>

<|im_start|>user
What is an IP address of the Google DNS servers?
<|im_end|>
<|im_start|>assistant
The IP address of a Google DNS server is

"""

unstructured = model(prompt, max_tokens=10)

structured = model(
    prompt,
    outlines.Regex(r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"),
    max_tokens=30
)

print(unstructured)
# 8.8.8.8
#
# <|im_end|>

print(structured)
# 8.8.8.8
```

Unlike other libraries, regex-structured generation in Outlines is almost as fast
as non-structured generation.

### Efficient JSON generation following a Pydantic model

Outlines users can guide the generation process so the output is *guaranteed* to follow a [JSON schema](https://json-schema.org/) or [Pydantic model](https://docs.pydantic.dev/latest/):

```python
from enum import Enum
from pydantic import BaseModel, constr
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines

class Weapon(str, Enum):
    sword = "sword"
    axe = "axe"
    mace = "mace"
    spear = "spear"
    bow = "bow"
    crossbow = "crossbow"


class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"


class Character(BaseModel):
    name: constr(max_length=10)
    age: int
    armor: Armor
    weapon: Weapon
    strength: int


model_name = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

# Construct structured sequence generator
generator = outlines.Generator(model, Character)

# Draw a sample
seed = 789001
character = generator("Give me a character description", seed=seed)

print(repr(character))
# Character(name='Anderson', age=28, armor=<Armor.chainmail: 'chainmail'>, weapon=<Weapon.sword: 'sword'>, strength=8)

prompt = "Give me an interesting character description"
character = model(prompt, Character, seed=seed)

print(repr(character))
# Character(name='Vivian Thr', age=44, armor=<Armor.plate: 'plate'>, weapon=<Weapon.crossbow: 'crossbow'>, strength=125)
```

The method works with union types, optional types, arrays, nested schemas, etc. Some field constraints are [not supported yet](https://github.com/dottxt-ai/outlines/issues/215), but everything else should work.

### Efficient JSON generation following a JSON Schema

Sometimes you just want to be able to pass a JSON Schema instead of a Pydantic model. We've got you covered:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines

schema = '''{
    "title": "Character",
    "type": "object",
    "properties": {
        "name": {
            "title": "Name",
            "maxLength": 10,
            "type": "string"
        },
        "age": {
            "title": "Age",
            "type": "integer"
        },
        "armor": {"$ref": "#/definitions/Armor"},
        "weapon": {"$ref": "#/definitions/Weapon"},
        "strength": {
            "title": "Strength",
            "type": "integer"
        }
    },
    "required": ["name", "age", "armor", "weapon", "strength"],
    "definitions": {
        "Armor": {
            "title": "Armor",
            "description": "An enumeration.",
            "enum": ["leather", "chainmail", "plate"],
            "type": "string"
        },
        "Weapon": {
            "title": "Weapon",
            "description": "An enumeration.",
            "enum": ["sword", "axe", "mace", "spear", "bow", "crossbow"],
            "type": "string"
        }
    }
}'''

model_name = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

prompt = "Give me a character description"
character = model(prompt, outlines.json_schema(schema))
```

### Using context-free grammars to guide generation

Formal grammars rule the world, and Outlines makes them rule LLMs too. You can pass any context-free grammar in the EBNF format and Outlines will generate an output that is valid to this grammar:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines

arithmetic_grammar = """
    ?start: expression

    ?expression: term (("+" | "-") term)*

    ?term: factor (("*" | "/") factor)*

    ?factor: NUMBER
           | "-" factor
           | "(" expression ")"

    %import common.NUMBER
"""

model_name = "WizardLM/WizardMath-7B-V1.1"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

prompt = "Alice had 4 apples and Bob ate 2. Write an expression for Alice's apples:"
sequence = model(prompt, outlines.types.Cfg(arithmetic_grammar))

print(sequence)
# (8-2)
```

This was a very simple grammar, and you can use `outlines.generate.cfg` to generate syntactically valid Python, SQL, and much more than this. Any kind of structured text, really. All you have to do is search for "X EBNF grammar" on the web, and take a look at the [Outlines `grammars` module](https://github.com/dottxt-ai/outlines/tree/main/outlines/grammars).

### Open functions

Outlines can infer the structure of the output from the signature of a function. The result is a dictionary, and can be passed directly to the function using the usual dictionary expansion syntax `**`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines


def add(a: int, b: int):
    return a + b

model_name = "WizardLM/WizardMath-7B-V1.1"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)
result = model(
    "Return json with two integers named a and b respectively. a is odd and b even.",
    add
)

print(add(**result))
# 3
```

A great advantage of passing functions directly to specify the structure is that the structure of the LLM will change with the function's definition. No need to change the code at several places!

You can also embed various functions into an enum to generate params:

```python
from enum import Enum
from functools import partial

from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines


def add(a: int, b: int) -> int:
    return a + b

def mul(c: float, d: float) -> float:
    return c * d

class Operation(Enum):
    add = partial(add)
    mul = partial(mul)

model_name = "WizardLM/WizardMath-7B-V1.1"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)
generator = outlines.Generator(model, Operation)
result = generator("Return json with two float named c and d respectively. c is negative and d greater than 1.0.")
#print(result)
# {'c': -3.14, 'd': 1.5}
```

## Prompting

Building prompts can get messy. **Outlines** makes it easier to write and manage
prompts by encapsulating templates inside "template functions". Template
functions use the Jinja2 templating engine to help build complex prompts in a
concise manner.

Template functions are created by loading a Jinja2 template from a text file.
Assume you have the following prompt template defined in `prompt.txt`:

``` text
You are a sentiment-labelling assistant.

{% for example in examples %}
{{ example[0] }} // {{ example[1] }}
{% endfor %}
{{ to_label }} //
```

You can then load it and call it with:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines

examples = [
    ("The food was disgusting", "Negative"),
    ("We had a fantastic night", "Positive"),
    ("Recommended", "Positive"),
    ("The waiter was rude", "Negative")
]

<<<<<<< HEAD
labelling = outlines.Template.from_file("prompt.txt")
prompt = labelling("Just awesome", examples)
```

<<<<<<< HEAD
This helps:

- Keep content separate from the code
- Design "white space perfect" prompts

It is more maintainable and means prompts can be versioned separately from the code.
=======
You can also load a template from a text file. Assume you have the following prompt template defined in `prompt.txt`:

``` text
You are a sentiment-labelling assistant.

{% for example in examples %}
{{ example[0] }} // {{ example[1] }}
{% endfor %}
{{ to_label }} //
```

You can load it with:

``` python
import outlines

=======
>>>>>>> 732de37 (Add deprecation warning for the `prompt` decorator)
labelling = outlines.Template.from_file("prompt.txt")
prompt = labelling("Just awesome", examples)
```
>>>>>>> 8b15cf3 (Rename `Prompt` to `Template`)

This helps:

- Keep content separate from the code
- Design "white space perfect" prompts

It is more maintainable and means prompts can be versioned separately from the code.

## Join us

- 💡 **Have an idea?** Come chat with us on [Discord][discord]
- 🔨 **Want to contribute?** Consult our [contribution guide](https://dottxt-ai.github.io/outlines/latest/community/contribute/).
- 🐞 **Found a bug?** Open an [issue](https://github.com/dottxt-ai/outlines/issues)


## Cite Outlines

```
@article{willard2023efficient,
  title={Efficient Guided Generation for LLMs},
  author={Willard, Brandon T and Louf, R{\'e}mi},
  journal={arXiv preprint arXiv:2307.09702},
  year={2023}
}
```

[documentation]: https://dottxt-ai.github.io/outlines/latest/welcome/
[documentation-badge]: https://img.shields.io/readthedocs/outlines
[contributors]: https://github.com/dottxt-ai/outlines/graphs/contributors
[contributors-badge]: https://img.shields.io/github/contributors/dottxt-ai/outlines?style=flat-square&logo=github&logoColor=white&color=ECEFF4
[dottxt-twitter]: https://twitter.com/dottxtai
[discord]: https://discord.gg/R9DSu34mGd
[discord-badge]: https://img.shields.io/discord/1182316225284554793?color=81A1C1&logo=discord&logoColor=white&style=flat-square
[downloads-badge]: https://img.shields.io/pypi/dm/outlines?color=89AC6B&logo=python&logoColor=white&style=flat-square
[pypistats]: https://pypistats.org/packages/outlines
[dottxt-twitter-badge]: https://img.shields.io/twitter/follow/dottxtai?style=social
[youtube-dottxt]: https://www.youtube.com/@dottxt-ai
[blog-dottxt]: https://blog.dottxt.co/
