from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines.models.transformers import from_transformers
from outlines.types import StructuralTags

MODEL_NAME = "erwanf/gpt2-mini"

model = from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

output_type = StructuralTags("""{
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<function="],
        "tags": [
            {
                "begin": "<function=get_weather>",
                "content": {
                    "type": "json_schema",
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "pattern": "[a-zA-Z]{1,20}"}
                        },
                        "required": ["location"]
                    }
                },
                "end": "</function>"
            },
            {
                "begin": "<function=get_time>",
                "content": {
                    "type": "json_schema",
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "pattern": "[a-zA-Z]{1,20}"}
                        },
                        "required": ["location"]
                    }
                },
                "end": "</function>"
            }
        ],
        "at_least_one": true,
        "stop_after_first": false
    }
}""")

prompt = f"""
You are a helpful assistant that can answer questions about the weather and the time.
You have access to the following functions:
- <function=get_weather>: parameters: location. Get the weather for a given location.
- <function=get_time>: parameters: location. Get the time for a given location.

What's the current time and weather in Tokyo? Call both the functions get_weather and get_time.
"""

result = model(prompt, output_type, max_new_tokens=100)
print(result)

#<function=get_time>{
#
#"location": "weather"
#
#}</function>
#
#You can also use the following functions to get the weather for the weather.
#
#You can also use the following functions to get the weather for the weather.
#
#You can also use the following functions to get the weather for the weather.
#
#You can also use the following functions to get the weather for the weather.
#
#You can also use the following functions to
