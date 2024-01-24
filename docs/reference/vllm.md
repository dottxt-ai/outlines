# Serve with vLLM

Outlines can be deployed as an LLM service using the vLLM inference engine and a FastAPI server. vLLM is not installed by default so will need to install Outlines with:

```bash
pip install outlines[serve]
```

You can then start the server with:

```bash
python -m outlines.serve.serve
```

This will by default start a server at `http://127.0.0.1:8000` (check what the console says, though) with the OPT-125M model. If you want to specify another model (e.g. Mistral-7B-Instruct-v0.2), you can do so with the `--model` parameter:

```bash
python -m outlines.serve.serve --model="mistralai/Mistral-7B-Instruct-v0.2"
```

You can then query the model in shell by passing a prompt and either

1. a [JSON Schema][jsonschema]{:target="_blank"} specification or
2. a [Regex][regex]{:target="_blank"} pattern

with the `schema` or `regex` parameters, respectively, to the `/generate` endpoint. If both are specified, the schema will be used. If neither is specified, the generated text will be unconstrained.

For example, to generate a string that matches the schema `{"type": "string"}` (any string):

```bash
curl http://127.0.0.1:8000/generate \
    -d '{
        "prompt": "What is the capital of France?",
        "schema": {"type": "string", "maxLength": 5}
        }'
```

To generate a string that matches the regex `(-)?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-][0-9]+)?` (a number):

```bash
curl http://127.0.0.1:8000/generate \
    -d '{
        "prompt": "What is Pi? Give me the first 15 digits: ",
        "regex": "(-)?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-][0-9]+)?"
        }'
```

Instead of `curl`, you can also use the [requests][requests]{:target="_blank"} library from another python program.

Please consult the [vLLM documentation][vllm]{:target="_blank"} for details on additional request parameters. You can also [read the code](https://github.com/outlines-dev/outlines/blob/main/outlines/serve/serve.py) in case you need to customize the solution to your needs.

[requests]: https://requests.readthedocs.io/en/latest/
[vllm]: https://docs.vllm.ai/en/latest/index.html
[jsonschema]: https://json-schema.org/learn/getting-started-step-by-step
[regex]: https://www.regular-expressions.info/tutorial.html
