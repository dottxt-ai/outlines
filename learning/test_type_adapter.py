"""Test LMStudio integration."""
import sys
sys.path.insert(0, "/Users/anrilombard/Desktop/outlines")

import lmstudio as lms
from outlines.models.lmstudio import LMStudioTypeAdapter, LMStudio, from_lmstudio
from outlines.inputs import Chat

adapter = LMStudioTypeAdapter()
client = lms.get_default_client()

# Test 1: String input
print("=== Test 1: String input ===")
result = adapter.format_input("Hello world")
print(f"Input: 'Hello world'")
print(f"Output: {result!r}")
print(f"Type: {type(result)}")

# Test 2: Chat input
print("\n=== Test 2: Chat input ===")
chat = Chat()
chat.add_system_message("You are helpful")
chat.add_user_message("What is 2+2?")
print(f"Input Chat messages: {chat.messages}")
result = adapter.format_input(chat)
print(f"Output type: {type(result)}")
print(f"Output: {result}")

# Test 3: Chat with conversation
print("\n=== Test 3: Multi-turn Chat ===")
chat2 = Chat()
chat2.add_system_message("Be brief")
chat2.add_user_message("Hi")
chat2.add_assistant_message("Hello!")
chat2.add_user_message("How are you?")
print(f"Input messages: {chat2.messages}")
result = adapter.format_input(chat2)
print(f"Output: {result}")

# Test 4: format_output_type with None
print("\n=== Test 4: format_output_type(None) ===")
result = adapter.format_output_type(None)
print(f"Output: {result}")

# Test 5: format_output_type with Pydantic
print("\n=== Test 5: format_output_type(Pydantic) ===")
from pydantic import BaseModel
class Person(BaseModel):
    name: str
    age: int

result = adapter.format_output_type(Person)
print(f"Output: {result}")

# Test 6: from_lmstudio factory
print("\n=== Test 6: from_lmstudio() ===")
model = from_lmstudio(client)
print(f"Model type: {type(model)}")
print(f"Has client: {hasattr(model, 'client')}")
print(f"Has model_name: {hasattr(model, 'model_name')}")
result = model.generate("Say 'hello' and nothing else.")
print(f"Result: {result!r}")
print(f"Type: {type(result)}")

# Test 7: LMStudio sync class - generate with structured output
print("\n=== Test 7: LMStudio.generate() with Pydantic ===")
result = model.generate("Create a person named Bob who is 25.", Person)
print(f"Result: {result!r}")
print(f"Type: {type(result)}")

# Test 8: LMStudio sync class - streaming
print("\n=== Test 8: LMStudio.generate_stream() ===")
print("Streaming: ", end="")
for chunk in model.generate_stream("Count from 1 to 3."):
    print(chunk, end="", flush=True)
print()

print("\n=== All tests passed ===")
