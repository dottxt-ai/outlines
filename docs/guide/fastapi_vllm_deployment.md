---
title: Deploying with FastAPI
---

# Deploying with FastAPI

This guide demonstrates how to build a FastAPI application that leverages Outlines' async integration with vLLM. We create a customer support API that can intelligently categorize tickets and generate structured responses.

## Prerequisites

Before starting, ensure you have a vLLM server running (locally or remotely) and the following packages installed:

```shell
pip install fastapi uvicorn outlines openai pydantic
```

## Building the Application

### Step 1: Define Data Models

First, let's define our Pydantic models for structured outputs:

```python
# models.py
from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    PRODUCT = "product"
    OTHER = "other"

class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TicketAnalysis(BaseModel):
    category: TicketCategory
    priority: TicketPriority
    summary: str = Field(description="Brief summary of the issue")
    customer_sentiment: str = Field(description="Customer emotional state")
    key_issues: List[str] = Field(description="List of main problems")
    requires_human: bool = Field(description="Whether this needs human intervention")

class SupportResponse(BaseModel):
    greeting: str
    acknowledgment: str = Field(description="Acknowledge the customer's issue")
    solution_steps: List[str] = Field(description="Steps to resolve the issue")
    closing: str
```

### Step 2: Define the prompts

Let us now write the prompts that we will be using in our application, using Jinja 2's templating language. We separate them from the application implementation so they are easier to modify and version.

```ascii
{# prompts/categorize.txt #}
Analyze this customer support ticket:

Customer ID: {{ customer_id }}
Message: {{ message }}

Extract the category, priority, and other relevant information.
```

```ascii
{# prompts/respond.txt #}
Generate a professional customer support response.

Customer Message: {{ message }}
Category: {{ category }}
Priority: {{  priority }}
Customer Sentiment: {{ customer_sentiment }}

Create a helpful, empathetic response that addresses their concerns.
```

### Step 3: Create the FastAPI Application

Now let's create our FastAPI application with async vLLM integration:

```python
# main.py
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import openai
from outlines import models, Template
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import TicketAnalysis, SupportResponse

# Request model
class TicketRequest(BaseModel):
    customer_id: str
    message: str

# Global model instance
async_model = None

# The lifespan function is a FastAPI construct
# used to define startup and shutdown logic for the API.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the async vLLM model on startup."""
    global async_model

    client = openai.AsyncOpenAI(
        base_url="http://localhost:8000/v1",  # Adjust to your vLLM server URL
        api_key="dummy"  # vLLM doesn't require a real API key
    )
    async_model = models.from_vllm(client, "Qwen/Qwen2.5-VL-7B-Instruct")

    yield

    async_model = None  # Cleanup

# Create FastAPI app
app = FastAPI(
    title="Customer Support Assistant API",
    description="AI-powered customer support with structured outputs",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/analyze-ticket", response_model=TicketAnalysis)
async def analyze_ticket(request: TicketRequest):
    """Analyze a customer support ticket and extract structured information."""
    if async_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    template = Template.from_file("prompts/categorize.txt")
    prompt = template(
        customer_id=request.customer_id,
        message=request.message
    )

    try:
        # Generate and parse a structured response
        result = await async_model(prompt, TicketAnalysis, max_tokens=5000)
        analysis = TicketAnalysis.model_validate_json(result)

        return analysis

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/generate-response", response_model=SupportResponse)
async def generate_response(
    request: TicketRequest,
    analysis: TicketAnalysis
):
    """Generate a structured support response based on ticket analysis."""
    if async_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    template = Template.from_file("prompts/respond.txt")
    prompt = template(
        message=request.message,
        category=analysis.category,
        priority=analysis.priority,
        customer_sentiment=analysis.customer_sentiment
    )

    try:
        # Generate and parse a structured response
        result = await async_model(prompt, SupportResponse, max_tokens=5000)
        response = SupportResponse.model_validate_json(result)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")
```

## Running the Application

### Step 1: Start your vLLM server

```shell
vllm serve Qwen/Qwen2.5-VL-7B-Instruct
```

### Step 2: Run the FastAPI application

```shell
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

## Testing the API

### Example 1: Analyze a support ticket

```shell
curl -X POST "http://localhost:8080/analyze-ticket" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST123",
    "message": "I have been charged twice for my subscription this month. This is unacceptable and I want a refund immediately!"
  }'
```

Expected response:

```json
{
  "category": "billing",
  "priority": "high",
  "summary": "Customer charged twice for subscription, requesting refund",
  "customer_sentiment": "angry",
  "key_issues": ["duplicate charge", "subscription billing", "refund request"],
  "requires_human": false
}
```

### Example 2: Generate a support response

```shell
# First, get the analysis
ANALYSIS=$(curl -s -X POST "http://localhost:8080/analyze-ticket" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST456",
    "message": "My app keeps crashing when I try to upload photos."
  }')

# Then generate a response
curl -X POST "http://localhost:8080/generate-response" \
  -H "Content-Type: application/json" \
  -d "{
    \"request\": {
      \"customer_id\": \"CUST456\",
      \"message\": \"My app keeps crashing when I try to upload photos.\"
    },
    \"analysis\": $ANALYSIS
  }"
```

By combining FastAPI's async capabilities with Outlines' structured generation, you can build robust APIs that leverage large language models.

## Using Alternative Backends: SGLang and TGI

One of the key advantages of Outlines is its unified API across different inference backends. You can easily switch from vLLM to SGLang or TGI with minimal code changes - just modify the model initialization in the `lifespan` function.

### Using SGLang Instead of vLLM

To use SGLang, simply change the client initialization:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the async SGLang model on startup."""
    global async_model

    client = openai.AsyncOpenAI(
        base_url="http://localhost:30000/v1",  # SGLang server URL
        api_key="dummy"
    )
    async_model = models.from_sglang(client)

    yield

    async_model = None
```

Start your SGLang server with:

```shell
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --port 30000
```

### Using TGI Instead of vLLM

For TGI (Text Generation Inference), use the Hugging Face client:

```python
import huggingface_hub

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the async TGI model on startup."""
    global async_model

    client = huggingface_hub.AsyncInferenceClient(
        "http://localhost:8080"  # TGI server URL
    )
    async_model = models.from_tgi(client)

    yield

    async_model = None
```

Start your TGI server with:

```shell
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf
```

The rest of your FastAPI application - all the endpoints, error handling, and business logic - remains completely unchanged. This flexibility allows you to test different inference engines without rewriting your application.
