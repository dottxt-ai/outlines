Controlled Generation
=====================

While LLM capabilities are increasingly impressive, we can make their output more reliable by *steering* the generation. Outlines thus offers mechanisms to specify high level constraints on text completions by generative language models.


Stopping sequence
-----------------

By default, language models stop generating tokens after and `<EOS>` token was generated, or after a set maximum number of tokens. Their output can be verbose, and for practical purposes it is often necessary to stop the generation after a given sequence has been found instead. You can use the `stop_at` keyword argument when calling the model with a prompt:

.. code::

   import outlines.models as models

   complete = models.text_completion.openai("text-davinci-002")
   expert = complete("Name an expert in quantum gravity.", stop_at=["\n", "."])


.. warning::

   The OpenAI API does not allow more than 4 stopping sequences.


Choice between different options
--------------------------------

In some cases we know the output is to be chosen between different options. We can restrict the completion's output to these choices using the `is_in` keyword argument:


.. code::


   import outlines.models as models

   complete = models.text_completion.openai("text-davinci-002")
   answer = model(
       "Pick the odd word out: skirt, dress, pen, jacket",
       is_in=["skirt", "dress", "pen", "jacket"]
   )


Type constraints
----------------

We can ask completions to be restricted to `int`s or `float`s using the `type` keyword argument, respectively with the "int" or "float" value:


.. code::


   import outlines.models as models

   complete = models.text_completion.openai("text-davinci-002")
   answer = model(
       "When I was 6 my sister was half my age. Now I’m 70 how old is my sister?",
       type="int"
   )


.. warning::

   This feature is very limited for OpenAI models, due to restrictions on OpenAI's API.


The future of constrained generation
------------------------------------

We believe constrained hold a lot of promises when it comes to build reliable systems that use language models. In future releases of Outlines, you will be able to:

- Exclude sequences with a `not_in` keyword agument;
- Constrain the output to be valid JSON;
- Constrain the output to be a valid array;
- Constrain the output to be valid Python code;

We also believe that `alternative steering methods <https://www.alignmentforum.org/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector>`_ can be useful and plan on expanding Outline's prompt DSL and generation methods in this direction.
