✨ Installation
===============

The latest version of outlines is available on PyPi:

.. code:: bash

   pip install outlines

Outlines comes with a minimal set of dependencies that are necessary to run the library's code. Integrations will require you to install dependencies manually.


OpenAI
------

To use OpenAI models you first have to run:

.. code:: bash

   pip install openai tiktoken

.. important::

    You also need to set your API credentials by defining the ``OPENAI_API_KEY`` environment variable.


HuggingFace
-----------

To use the integrations with HuggingFace's `transformers <https://huggingface.co/docs/transformers/index>`_ and `diffusers <https://huggingface.co/docs/diffusers/index>`_ libraries you first need to run:

.. code::

   pip install torch transformers diffusers


.. attention::

   HuggingFace models are run locally. Outlines uses the `PyTorch <https://pytorch.org/>`_ versions of the models. Please refer to the `PyTorch documentation <https://pytorch.org/get-started/locally/>`_ for questions related to **GPU support**.

The integration is fairly basic for now, and if you have specific performance needs please `open an issue <https://github.com/normal-computing/outlines/issues>`_

Other integrations
------------------

Outlines is designed to be fully compatible with other libraries, which you will need to install separately. You can use any library with Outlines but , whenever possible, we recommend to use libraries with async support for better performance. Examples of possible integrations are:

- `Llama index <https://github.com/jerryjliu/llama_index>`_ for vector stores and document querying;
- `discord.py <https://discordpy.readthedocs.io/en/stable/index.html>`_ for Discord integration;
- `Slack SDK <https://slack.dev/python-slack-sdk/>`_ for Slack integration;
- `aiofiles <https://github.com/Tinche/aiofiles>`_ for asynchronous file operations;
- `httpx <https://www.python-httpx.org/async/>`_ or `aiohttp <https://github.com/aio-libs/aiohttp>`_ for asynchronous HTTP requests;
- `asyncpg <https://github.com/MagicStack/asyncpg>`_ and `aiosqlite <https://github.com/omnilib/aiosqlite>`_ for async PostgreSQL and SQLite interfaces.
