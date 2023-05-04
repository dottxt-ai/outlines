🌎 Hello world
==============

Here is a simple Outlines program that highlights some of its key features:

.. code::

   import outlines.text as text
   import outlines.models as models


   @text.prompt
   def where_from(expression):
        "What's the origin of '{{ expression }}'?"


   complete = models.text_completion.openai("text-davinci-003")

   hello_world = where_from("Hello world")
   foobar = where_from("Foo Bar")
   answer = complete([hello_world, foobar], num_samples=3, stop_at=["."])


- **Prompt management**. You can use functions with the ``@outlines.text.prompt`` decorator. "Prompt functions" use the `Jinja templating language <https://jinja.palletsprojects.com/en/3.1.x/>`_ to render the prompt written in the docstring. We also added a few filters to help with common worflows, like building agents. Of course, for simple prompts, you can also use Python strings directly.
- **Generative model integration**. You can use text completion models from OpenAI and HuggingFace, but models are not limited to text.
- **Controlled generation**. The ``stop_at`` keyword arguments allows to define when the generation should be stopped. Outlines includes more options to control the generation; these happen on a token basis, saving time and costs.
- **Sampling**. Outlines exclusively generates sequences using sampling. You can generate many samples with one call.
- **Batching**. Models can take a list of prompt as input and generate completions in parallel.
