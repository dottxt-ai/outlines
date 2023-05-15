Multimodal, Multimodels
=======================

Outlines interfaces with multiple model providers, so models can be easily swapped. It is built so that different models can be chained together, with different modalities.

OpenAI
------

Outlines connects to OpenAI's text completion, and chat completion. Note however that Outlines does not provide a chat interface, and uses the chat completion API for text completion. Both are accessible via the `models.text_completion.openai` module, by passing the name of the model. You can currently specify `max_tokens` and `temperature` when initializing the model:

.. code::

   import outlines.models as models

   complete = models.text_completion.openai("gpt4", max_tokens=128, temperature=0.7)


It is also possible to use DALL-E to generate images:

.. code::

   import outlines.models as models

   generate = models.image_generation.openai("dall-e")


HuggingFace
-----------

Outlines can call models from HuggingFace's `transformers` and `diffusers` libraries. The models are then run locally.

.. code::

   import outlines.models as models

   complete = models.text_completion.hf("sshleifer/tiny-gpt2")
   generate = models.image_generation.hf("runwayml/stable-diffusion-v1-5")


.. note::

   Outlines call the PyTorch version of models by default. The generation process also runs with defaults, please `open an issue <https://github.com/normal-computing/outlines/issues>`_ if you have more specific needs.


Bring Your Own Model
--------------------

Outlines models are currently simple functions that return a text or an image given a prompt, you can thus easily use any model. We will soon provide a more comprehensive integration that handles controlled generation for any model.

If you think the model you are using could be useful to others, `open an issue <https://github.com/normal-computing/outlines/issues>`_ 😊


Coming soon
-----------

We plan on integrating more model providers, for instance:

- Anthropic
- Llamacpp
- GPT4All

We currently favor the integration of *Open Source models* since they give more freedom for guided generation. We will also extend the range of models to allow building more complex chains, including:

- Image captioning;
- Classification;
- Image segmentation;
- Speech-to-text;
- Image question answering;
- etc.
