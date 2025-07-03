# user provides Chat object
import io
import requests
import PIL
import openai
import outlines

from outlines.inputs import Chat, Image

model = outlines.from_openai(
    openai.OpenAI(),
    "gpt-4o"
)

def get_image(url):
    r = requests.get(url)
    return PIL.Image.open(io.BytesIO(r.content))

# user provides Chat object
prompt = Chat([
    ({"role": "system", "content": "You are a helpful assistant. Answer in French."}),
    ({"role": "user", "content": "Just a random message, ignore it."}),
    ({"role": "user", "content": "Describe this image", "items": [Image(get_image("https://picsum.photos/id/237/400/300"))]})
])
print(model(prompt, max_tokens=20))

# user provides a string with chat tags/messages
template_string = """
{%system%} You are a helpful assistant. Answer in French.{%endsystem%} 
{%user%} Just a random message, ignore it.{%enduser%} 
{%user%} {{prompt}} {%enduser%}
"""
template = outlines.templates.Template.from_string(template_string)
print(model(template(prompt="How are you?"), max_tokens=20))

# direct use of Vision (no chat)
print(model("Describe this image", items=[Image(get_image("https://picsum.photos/id/237/400/300"))], max_tokens=20))

# example of how it would work with tools (not implemented yet)
#def get_weather(city):
#    return {"temperature": 25, "condition": "sunny"}
#
#def get_news(topic):
#    return {"news": "The weather in Paris is sunny and 25 degrees Celsius."}
#
#prompt = Chat([
#    ({"role": "system", "content": "You are a helpful assistant. Answer in French. Use the tools provided to answer the user's question."}),
#    ({"role": "user", "content": "Tell me about the weather in Paris."}),
#    ({"role": "assistant", "content": "{'tool': 'get_weather', 'args': {'city': 'Paris'}}"}),
#    ({"role": "function", "content": "{'temperature': 25, 'condition': 'sunny'}"})
#])
#print(model(prompt, max_tokens=20, tools=[get_weather, get_news]))
