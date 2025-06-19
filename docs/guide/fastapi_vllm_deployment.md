---
title: Deploying with FastAPI
---

# Deploying with FastAPI

This guide demonstrates how to build a FastAPI application that leverages Outlines' async integration with vLLM. We'll create a customer support API that can intelligently categorize tickets and generate structured responses.

## Prerequisites

Before starting, ensure you have:

1. A vLLM server running (locally or remotely)
2. Python 3.8+ installed
3. The following packages installed:

```bash
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

### Step 2: Create the FastAPI Application

Now let's create our FastAPI application with async vLLM integration:

```python
# main.py
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import openai
import outlines
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import TicketAnalysis, SupportResponse

# Request model
class TicketRequest(BaseModel):
    customer_id: str
    message: str

# Global model instance
async_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the async vLLM model on startup."""
    global async_model

    client = openai.AsyncOpenAI(
        base_url="http://localhost:8000/v1",  # Adjust to your vLLM server URL
        api_key="dummy"  # vLLM doesn't require a real API key
    )
    async_model = outlines.from_vllm(client)

    yield

    async_model = None  # Cleanup

# Create FastAPI app
app = FastAPI(
    title="Customer Support Assistant API",
    description="AI-powered customer support with structured outputs",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Check if the service is running and vLLM is accessible."""
    if async_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy"}


@app.post("/analyze-ticket", response_model=TicketAnalysis)
async def analyze_ticket(request: TicketRequest):
    """Analyze a customer support ticket and extract structured information."""
    if async_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    prompt = f"""Analyze this customer support ticket:

Customer ID: {request.customer_id}
Message: {request.message}

Extract the category, priority, and other relevant information."""

    try:
        # Use async model with structured output
        result = await async_model(
            prompt,
            output_type=TicketAnalysis,
            max_tokens=500,
            temperature=0.1
        )

        # Parse and return the result
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

    prompt = f"""Generate a professional customer support response.

Customer Message: {request.message}
Category: {analysis.category}
Priority: {analysis.priority}
Customer Sentiment: {analysis.customer_sentiment}

Create a helpful, empathetic response that addresses their concerns."""

    try:
        # Generate structured response
        result = await async_model(
            prompt,
            output_type=SupportResponse,
            max_tokens=800,
            temperature=0.7
        )

        response = SupportResponse.model_validate_json(result)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")
```

## Running the Application

### Step 1: Start your vLLM server

```bash
# Example: Start vLLM with a model
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000
```

### Step 2: Run the FastAPI application

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Production with multiple workers
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8080
```

## Testing the API

### Example 1: Analyze a support ticket

```bash
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

```bash
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

By combining FastAPI's async capabilities with Outlines' structured generation, you can build robust APIs that leverage large language models while maintaining high performance and reliability.

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
    async_model = outlines.from_sglang(client)

    yield

    async_model = None
```

Start your SGLang server with:
```bash
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
    async_model = outlines.from_tgi(client)

    yield

    async_model = None
```

Start your TGI server with:
```bash
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf
```

The rest of your FastAPI application - all the endpoints, error handling, and business logic - remains completely unchanged. This flexibility allows you to test different inference engines without rewriting your application.
