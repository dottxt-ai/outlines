---
title: Skeleton-of-Thought
---

# Skeleton-of-Thought


[Skeleton-of-Thought](http://arxiv.org/abs/2307.15337) is a prompting technique that breaks down complex reasoning tasks into a high-level outline or "skeleton" of key steps, before filling in the details. It involves first generating a basic structure or framework for approaching the problem, then expanding on each step to arrive at a full solution. This technique helps guide the language model's reasoning process in a more structured and organized way, especially for multi-step problems.


## Code Example


```python
from pydantic import BaseModel, Field
from typing import List
import outlines

class Step(BaseModel):
    title: str
    description: str = ""

class Skeleton(BaseModel):
    steps: List[Step]

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

# Generate the skeleton
skeleton_generator = outlines.generate.json(model, Skeleton)
skeleton = skeleton_generator(
    "Create a skeleton outline for developing a new user authentication feature for a web application."
)

print("Skeleton:")
for i, step in enumerate(skeleton.steps, 1):
    print(f"{i}. {step.title}")

# Expand each step
step_expander = outlines.generate.json(model, Step)

expanded_steps = []
for step in skeleton.steps:
    expanded_step = step_expander(
        f"Expand on the following step for developing a user authentication feature: {step.title}"
    )
    expanded_steps.append(expanded_step)

print("\nExpanded Steps:")
for i, step in enumerate(expanded_steps, 1):
    print(f"{i}. {step.title}")
    print(f"   {step.description}\n")
```



    Skeleton:
    1. Determine Authentication Requirements
    2. Set up Authentication Mechanisms
    3. Design User Interface
    4. Implement Authentication Process
    5. Test Authentication Functionality
    6. Roll Out New Feature


    Compiling FSM index for all state transitions: 100%|█| 48/48 [00:00<00:00,



    Expanded Steps:
    1. Determine Authentication Requirements
       In order to develop a user authentication feature, determine your authentication requirements. Do you want to authenticate users via email password or username password? And what types of users will your authentication system handle: registered users, guests or both?",

    2. Step 2
       Develop a user login page and implement the necessary authentication mechanisms. This can include CAPTCHAS, multi-factor authentication, two-factor authentication, or other similar methods.

    3. User Authentication
       Card Layout Design for User Authentication feature. First step in design for the user interface of this feature is the card layout design. This design will allow the user to provide the necessary details for authentication in a structured way. The card layout design consists of boxes with labels for each of the essential details such as username, password, email address and security questions. These boxes should be of sufficient size to match the average user's fingerprint

    4. Implementing a Strong Authentication Process for User Protection


    5. Test Authentication Functionality
       After implementing authentication, test the feature to ensure that it is working as intended. Run a complete suite of tests covering all possible authentication scenarios, including login, registration, password expiration, two-factor authentication, and other authenticators such as biometric or one-time password. Validate the outputs of each test case by comparing the result against the expected output, and ensure any bugs or failures are properly tracked and resolved. Verify that all security protocols are in place, such as encryption and secure communication protocols, and ensure that user data is properly stored and managed. Finally, perform penetration testing to identify any potential weaknesses and regulations compliance.

    6. User Authentication
       The new feature will allow users to create and log into their accounts, providing them with access to premium content and better user experience. The feature will use OAuth for authentication and lead the user to a secure login page on our site.


We can see that this model does a relatively good job, but that we might want to combine this technique with another to improve the quality of the output.
