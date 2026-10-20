"""Exercise MiniMax through the real SDK with an offline HTTP transport."""

import io
import json

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI
from PIL import Image as PILImage

import outlines
from outlines.inputs import Image, Video
from outlines.models.minimax import MINIMAX_BASE_URLS


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("with_media", [False, True])
@pytest.mark.parametrize("base_url", MINIMAX_BASE_URLS.values())
@pytest.mark.parametrize(
    "model_name,kwargs,expected_model",
    [
        (None, {}, "MiniMax-M3.1-Flash"),
        ("MiniMax-M3", {}, "MiniMax-M3"),
        ("MiniMax-M2.7", {}, "MiniMax-M2.7"),
        ("custom-model", {}, "custom-model"),
        (None, {"model": "MiniMax-M2.7-highspeed"}, "MiniMax-M2.7-highspeed"),
        ("MiniMax-M3", {"model": "custom-model"}, "custom-model"),
        (None, {"reasoning_effort": "low"}, "MiniMax-M3.1-Flash"),
        (None, {"reasoning_effort": "medium"}, "MiniMax-M3.1-Flash"),
        (None, {"reasoning_effort": "high"}, "MiniMax-M3.1-Flash"),
        (None, {"reasoning_effort": "xhigh"}, "MiniMax-M3.1-Flash"),
        (None, {"reasoning_effort": "max"}, "MiniMax-M3.1-Flash"),
        (
            "MiniMax-M3",
            {"extra_body": {"thinking": {"type": "disabled"}}},
            "MiniMax-M3",
        ),
        (
            None,
            {"extra_body": {"thinking": {"type": "adaptive"}}},
            "MiniMax-M3.1-Flash",
        ),
    ],
)
async def test_minimax_sdk_payload(
    async_client, stream, with_media, base_url, model_name, kwargs, expected_model
):
    requests = []

    def handle(request):
        requests.append(request)
        if stream:
            chunk = {
                "id": "offline",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": expected_model,
                "choices": [{"index": 0, "delta": {"content": "ok"}}],
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": expected_model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            },
        )

    transport = httpx.MockTransport(handle)
    if async_client:
        client = AsyncOpenAI(
            api_key="offline-test", base_url=base_url,
            http_client=httpx.AsyncClient(transport=transport),
        )
    else:
        client = OpenAI(
            api_key="offline-test", base_url=base_url,
            http_client=httpx.Client(transport=transport),
        )
    if model_name is None:
        model = outlines.from_minimax(client)
    else:
        model = outlines.from_minimax(client, model_name)
    buffer = io.BytesIO()
    PILImage.new("RGB", (1, 1)).save(buffer, format="PNG")
    buffer.seek(0)
    image = Image(PILImage.open(buffer))
    video = Video("https://example.com/clip.mp4")
    prompt = ["Describe these assets", image, video] if with_media else "Hello"
    try:
        if stream:
            if async_client:
                result = "".join([part async for part in model.stream(prompt, **kwargs)])
            else:
                result = "".join(model.stream(prompt, **kwargs))
        elif async_client:
            result = await model(prompt, **kwargs)
        else:
            result = model(prompt, **kwargs)
        assert result == "ok"
    finally:
        if async_client:
            await client.close()
        else:
            client.close()

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{base_url}/chat/completions"
    expected = {
        "model": expected_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe these assets"},
                {"type": "image_url", "image_url": {
                    "url": f"data:{image.image_format};base64,{image.image_str}"
                }},
                {"type": "video_url", "video_url": {"url": video.video}},
            ],
        }],
    }
    if not with_media:
        expected["messages"] = [{"role": "user", "content": "Hello"}]
    expected.update({key: value for key, value in kwargs.items() if key != "extra_body"})
    expected.update(kwargs.get("extra_body", {}))
    if stream:
        expected["stream"] = True
    assert json.loads(request.content) == expected
