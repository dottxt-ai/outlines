"""
Verify LMStudio SDK behavior before building Outlines integration.
Run with LMStudio app open and a model loaded.

Usage:
    python learning/test_lmstudio_sdk.py
"""
import lmstudio as lms
from pydantic import BaseModel


class Book(BaseModel):
    title: str
    author: str
    year: int


def test_basic_generation():
    """Test basic text generation and response type."""
    print("\n=== Test 1: Basic Generation ===")
    model = lms.llm()

    # Test respond() with string
    result = model.respond("Say 'hello' and nothing else.")

    print(f"Response type: {type(result)}")
    print(f"result.content: {result.content!r}")
    print(f"result.content type: {type(result.content)}")

    # Check available attributes
    print(f"Has .parsed: {hasattr(result, 'parsed')}")
    print(f"Has .stats: {hasattr(result, 'stats')}")
    print(f"Has .model_info: {hasattr(result, 'model_info')}")

    return result


def test_structured_pydantic():
    """Test structured output with Pydantic model."""
    print("\n=== Test 2: Structured Output (Pydantic) ===")
    model = lms.llm()

    result = model.respond(
        "Tell me about The Hobbit book. Return JSON.",
        response_format=Book
    )

    print(f"result.content: {result.content!r}")
    print(f"result.parsed: {result.parsed}")
    print(f"result.parsed type: {type(result.parsed)}")

    # Verify it's a dict, not Pydantic instance
    if isinstance(result.parsed, dict):
        print("CONFIRMED: result.parsed is a dict")
        # Convert to Pydantic
        book = Book.model_validate(result.parsed)
        print(f"Converted to Pydantic: {book}")

    return result


def test_structured_json_schema():
    """Test structured output with raw JSON schema dict."""
    print("\n=== Test 3: Structured Output (JSON Schema Dict) ===")
    model = lms.llm()

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }

    result = model.respond(
        "Create a person named Alice who is 30 years old. Return JSON.",
        response_format=schema
    )

    print(f"result.content: {result.content!r}")
    print(f"result.parsed: {result.parsed}")
    print(f"result.parsed type: {type(result.parsed)}")

    return result


def test_streaming():
    """Test streaming and fragment structure."""
    print("\n=== Test 4: Streaming ===")
    model = lms.llm()

    stream = model.respond_stream("Count from 1 to 5.")

    print("Streaming fragments:")
    fragments = []
    for i, fragment in enumerate(stream):
        fragments.append(fragment)
        print(f"  Fragment {i}: content={fragment.content!r}, type={type(fragment)}")
        if i == 0:
            # Inspect first fragment attributes
            print(f"  Fragment attrs: {[a for a in dir(fragment) if not a.startswith('_')]}")

    # Get final result
    result = stream.result()
    print(f"\nFinal result.content: {result.content!r}")

    return fragments, result


def test_chat():
    """Test Chat class."""
    print("\n=== Test 5: Chat Context ===")
    model = lms.llm()

    chat = lms.Chat("You are a helpful assistant who responds briefly.")
    chat.add_user_message("What is 2+2?")

    print(f"Chat type: {type(chat)}")
    print(f"Chat messages: {chat}")

    result = model.respond(chat)
    print(f"Response: {result.content!r}")

    # Add response to chat - must add as assistant response, not PredictionResult
    chat.add_assistant_response(result.content)
    print(f"Chat after adding response: {chat}")

    return chat, result


def test_async():
    """Test async API."""
    print("\n=== Test 6: Async API ===")
    import asyncio

    async def async_test():
        async with lms.AsyncClient() as client:
            model = await client.llm.model()
            print(f"Async model type: {type(model)}")

            result = await model.respond("Say 'async works'")
            print(f"Async result.content: {result.content!r}")

            return result

    return asyncio.run(async_test())


def main():
    print("LMStudio SDK Verification Script")
    print("=" * 50)
    print("Ensure LMStudio is running with a model loaded.\n")

    try:
        test_basic_generation()
        test_structured_pydantic()
        test_structured_json_schema()
        test_streaming()
        test_chat()
        test_async()

        print("\n" + "=" * 50)
        print("All tests completed!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
