# user provides Chat object
import io
import requests
import PIL
import openai
import outlines

from outlines.inputs import Chat, Vision

model = outlines.from_openai(
    openai.OpenAI(),
    "gpt-4o"
)

def get_image(url):
    r = requests.get(url)
    return PIL.Image.open(io.BytesIO(r.content))

# user provides Chat object
prompt = Chat([
    ("system", "You are a helpful assistant. Answer in French."),
    ("user", "Just a random message, ignore it."),
    ("user", Vision(get_image("https://picsum.photos/id/237/400/300"), "Describe this image"))
])
print(model(prompt, max_tokens=20))

# user provides a string with chat tags/messages
prompt = """
{%system%} You are a helpful assistant. Answer in French.{%endsystem%} 
{%user%} Just a random message, ignore it.{%enduser%} 
{%user%} How are you doing?{%enduser%}
"""
print(model(prompt, max_tokens=20))

# direct use of Vision (no chat)
print(model(Vision(get_image("https://picsum.photos/id/237/400/300"), "Describe this image"), max_tokens=20))

# example of how it would work with tools (not implemented yet)
def get_weather(city):
    return {"temperature": 25, "condition": "sunny"}

def get_news(topic):
    return {"news": "The weather in Paris is sunny and 25 degrees Celsius."}

prompt = Chat([
    ("system", "You are a helpful assistant. Answer in French. Use the tools provided to answer the user's question."),
    ("user", "Tell me about the weather in Paris."),
    ("assistant", "{'tool': 'get_weather', 'args': {'city': 'Paris'}}"),
    ("function", "{'temperature': 25, 'condition': 'sunny'}")
])
print(model(prompt, max_tokens=20, tools=[get_weather, get_news]))
