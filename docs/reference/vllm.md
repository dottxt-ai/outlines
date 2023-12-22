# Serve with vLLM

Outlines can be deployed as an LLM service using the vLLM inference engine and a FastAPI server. vLLM is not installed by default so will need to install Outlines with:

```bash
pip install outlines[serve]
```

You can then start the server with:

```python
python -m outlines.serve.serve
```

This will by default start a server at `http://127.0.0.1:8000` (check what the console says, though)  with the OPT-125M model. If you want to specify another model:

```python
python -m outlines.serve.serve --model="mistralai/Mistral-7B-v0.1"
```

You can then query the model in shell by passing a prompt and a [JSON Schema][jsonschema]{:target="_blank"} specification for the structure of the output:

```bash
curl http://0.0.0.1:8000 \
    -d '{
        "prompt": "What is the capital of France?",
        "schema": {"type": "string"}
        }'
```

Or use the [requests][requests]{:target="_blank"} library from another python program. You can read the [vLLM documentation][vllm]{:target="_blank"} for more details.

You can also [read the code](https://github.com/outlines-dev/outlines/blob/main/outlines/serve/serve.py) in case you need to customize the solution to your needs.

[requests]: https://requests.readthedocs.io/en/latest/
[vllm]: https://docs.vllm.ai/en/latest/index.html
