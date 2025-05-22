---
title: Application API
---

# Application

The `Application` class enables you to encapsulate a prompt template and an output type into a reusable component.

## Overview

An `Application` combines a prompt template with an output type, creating a reusable component that can be applied to different models.

## Basic Usage

```python
from outlines import Application, Template, from_transformers
from typing import List

# Create a template
template = Template("""
You are an expert classifier.
Classify the following text into one of these categories: {categories}

Text: {text}
""")

# Create an application with the template and output type
classifier = Application(template, str)

# Initialize a model
model = from_transformers(...)

# Use the application with specific parameters
categories = ["Sports", "Politics", "Technology", "Entertainment"]
result = classifier(model, categories=categories, text="Apple unveils new iPhone with AI capabilities")

print(result)  # Expected output: "Technology"
```

## Structured Output

```python
from pydantic import BaseModel
from outlines import Application, Template, from_transformers

# Define a Pydantic model for structured output
class MovieReview(BaseModel):
    title: str
    director: str
    year: int
    rating: float
    review: str

# Create a template
template = Template("""
Write a brief review of the movie {movie_title}.
Include the director, year of release, and a rating out of 10.
""")

# Create an application with the template and output type
movie_reviewer = Application(template, MovieReview)

# Use the application with a model
result = movie_reviewer(model, movie_title="The Matrix")

# Parse the result into a Pydantic model
review = MovieReview.model_validate_json(result)
print(f"{review.title} ({review.year}) - {review.rating}/10")
```

## Parameters

- `prompt_template`: A template that defines the prompt structure
- `output_type`: The type of output to generate

::: outlines.applications.Application
