# Chat templating

Instruction-tuned language models use "special tokens" to indicate different parts of text, such as the system prompt, the user prompt, any images, and the assistant's response. A [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating) is how different types of input are composited together into a single, machine-readable string.

Outlines does not manage chat templating tokens when using instruct models. You must apply the chat template tokens to the prompt yourself -- if you do not apply chat templating on instruction-tuned models, you will often get nonsensical output from the model.

Chat template tokens are not needed for base models.

You can find the chat template tokens in the model's HuggingFace repo or documentation. As an example, the `SmolLM2-360M-Instruct` special tokens can be found [here](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/main/special_tokens_map.json).

However, it can be slow to manually look up a model's special tokens, and special tokens vary by models. If you change the model, your prompts may break if you have hard-coded special tokens.

If you need a convenient tool to apply chat templating for you, you should use the `tokenizer` from the `transformers` library:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You extract information from text."},
        {"role": "user", "content": "What food does the following text describe?"},
    ],
    tokenize=False,
    add_bos=True,
    add_generation_prompt=True,
)
```

yields

```ascii
<|im_start|>system
You extract information from text.<|im_end|>
<|im_start|>user
What food does the following text describe?<|im_end|>
<|im_start|>assistant
```
