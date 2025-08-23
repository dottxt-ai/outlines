from outlines.models import from_openai
from openai import OpenAI

client = OpenAI()

model = from_openai(
    OpenAI(),
    "gpt-4o-mini",
)

tool = {
    "name": "get_current_time",
    "description": "Get the current time",
    "parameters": {
        "city": "string",
    },
    "required": ["city"],
}

result = model(
    "What is the current time in Tokyo?",
    tools=[tool],
)

print(result)
# {'content': None, 'type': 'tool_call', 'tool_calls': [{'name': 'get_current_time', 'args': '{"city":"Tokyo"}', 'tool_call_id': 'call_B0gcCjHtA54GxcZuOAggAu6v'}]}
