.. Outlines documentation master file, created by
   sphinx-quickstart on Thu May  4 11:16:27 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


👋 Welcome to Outlines
======================

**Outlines** is a Python library to write reliable programs for interactions with generative models: language models, diffusers, multimodal models, classifiers, etc. It provides a Domain Specific Language (DSL) to make prompting easier, constrained text generation and is natively concurrent. It integrates well with the rest of the Python ecosystem: tools, vector stores, etc.

*Outlines aims to be the library frameworks are made with. It is more like NumPy than LangChain.*

.. grid:: 2

   .. grid-item-card:: 💻 Install Outlines
      :link: https://pypi.org/project/outlines
      :text-align: center
      :width: 75%
      :margin: 4 4 auto auto

      .. code::

         pip install outlines

   .. grid-item-card:: 🚀 Normal Computing
      :link: https://normalcomputing.ai
      :text-align: center
      :width: 75%
      :margin: 4 4 auto auto

      The development of Outlines is entirely funded by `Normal Computing <https://normalcomputing.ai>`_


👀 Sneak Peek
-------------

A toy implementation of an agent (similar to BabyAGI or AutoGPT) with Outlines:

.. code:: python

   import outlines.text as text
   import outlines.models as models

   from my_tools import google_search, execute_code
   from my_response_models import command_response


   @text.prompt
   def agent_prompt(objective, goals, tools, response_model):
       """You are an AI with the following objective: {{ objective }}

       Keep the following goals in mind:
       {% for goal in goals %}
       {{ loop.counter }}. {{ goal }}
       {% endfor %}

       COMMANDS
       {% for tool in tools %}
       - {{ tool | name }}, {{ tool | description }}, {{ tool | signature }}
       {% endfor %}

       OUTPUT FORMAT
       {{ response_model | schema }}
       """


   def agent(objective, goals, tools)
      complete = models.text_completion.hf("sshleifer/tiny-gpt2")
      prompt = agent_prompt(objective, goals, tools , command_response)
      answer = complete(prompt)
      command = command_response(answer)

      return command


   agent(
       "Write a library called Outlines",
       ["Easy prompting", "Multimodal, multimodel", "Constrained text generation"],
       [google_search, execute_code],
   )

📜 Features
-----------
 Simple and powerful prompting primitives based on the Jinja templating engine.
 Integration with OpenAI and HuggingFace models

- A powerful domain-specific language to write and render prompts;
- Interleave completions with loops, conditionals, and custom Python functions;
- OpenAI integration: language models, embeddings and Dall-E;
- HuggingFace integration: ``transformers`` and ``diffusers``;
- Caching;
- Sampling multiple sequences;
- Controlled generation, including multiple choice, type constraints and dynamic stopping.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   overview

.. toctree::
   :maxdepth: 1
   :caption: Outlines
   :hidden:

   reference/prompting
   reference/controlled_generation
   reference/multimodel
   reference/batching
