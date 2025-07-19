# Chat templating

Instruction-tuned language models use "special tokens" to indicate different parts of text, such as the system prompt, the user prompt, any images, and the assistant's response. A [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating) is how different types of input are composited together into a single, machine-readable string.

Outlines supports chat templating throught the `Chat` model input class. It contains a list of messages similar in format to the chat history you would use with API models such as OpenAI or Anthropic and to the expected arguments of the `apply_chat_template` method of transformers tokenizers. You can find detailed information on the interface of this object in the [model inputs documentation](../features/core/inputs.md).
