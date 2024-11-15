This recipe demonstrates how to use the `outlines` library to extract structured event details from a text message.
We will extract the title, location, and start date and time from messages like the following:

```plaintext
Hello Kitty, my grandmother will be here, I think it's better to postpone
our appointment to review math lessons to next Monday at 2pm at the same
place, 3 avenue des tanneurs, one hour will be enough see you 😘
```

Let see how to extract the event details from the message.

```python

from datetime import datetime
from pydantic import Field, BaseModel
from outlines import models, generate

# Load the model
model = models.mlxlm("mlx-community/Hermes-3-Llama-3.1-8B-8bit")

# Define the event schema using Pydantic
class Event(BaseModel):
    title: str = Field(description="title of the event")
    location: str
    start: datetime = Field(default=None, description="date of the event if available in iso format")

# Get the current date and time
now = datetime.now().strftime("%d/%m/%Y %H:%M")

# Define the prompt
prompt = f"""
Today's date and time are {datetime.now().strftime("%d/%m/%Y %H:%M")}
Given a user message, extract information of the event like date and time in iso format, location and title.
Here is the message:
"""

# Sample message
message = """Hello Kitty, my grandmother will be here , I think it's better to postpone our
appointment to review math lessons to next Monday at 2pm at the same place, 3 avenue des tanneurs, I think that one hour will be enough
see you 😘 """

# Create the generator
generator = generate.json(model, Event)

# Extract the event information
event = generator(prompt + message)

# Print the current date and time
print("Today's date and time are", now)

# Print the extracted event information in JSON format
print(event.json())

```

The output will be:

```plaintext
Today's date and time are 15/11/2024 15:52
```

and the extracted event information will be:

```json
{
  "title":"Math lessons",
  "location":"3 avenue des tanneurs",
  "start":"2024-11-18T14:00:00Z"
}
```
