from datetime import datetime

from mlx_lm import load
from pydantic import BaseModel, Field

import outlines
from outlines.generate import Generator
from outlines.types import JsonType


# Load the model
model = outlines.from_mlxlm(*load("mlx-community/Hermes-3-Llama-3.1-8B-8bit"))


# Define the event schema using Pydantic
class Event(BaseModel):
    title: str = Field(description="title of the event")
    location: str
    start: datetime = Field(
        default=None, description="date of the event if available in iso format"
    )


# Get the current date and time
now = datetime.now().strftime("%A %d %B %Y and it's %H:%M")

# Define the prompt
prompt = f"""
Today's date and time are {now}
Given a user message, extract information of the event like date and time in iso format, location and title.
If the given date is relative, think step by step to find the right date.
Here is the message:
"""

# Sample message
message = """Hello Kitty, my grandmother will be here , I think it's better to postpone our
appointment to review math lessons to next Friday at 2pm at the same place, 3 avenue des tanneurs, I think that one hour will be enough
see you 😘 """

# Create the generator
generator = Generator(model, JsonType(Event))

# Extract the event information
event = generator(prompt + message)

# Print the current date and time
print(f"Today: {now}")

# Print the extracted event information in JSON format
print(event.json())
