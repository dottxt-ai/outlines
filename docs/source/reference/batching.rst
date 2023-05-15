Sampling
========

Outlines is sampling-first, and is built to generate several samples from the same prompt:

.. code::

   import outlines.models as models

   sample = models.text_generation.openai("text-davinci-003")
   answers = complete(
       "When I was 6 my sister was half my age. Now I’m 70 how old is my sister?",
       samples=10
    )

This will enable probabilistic applications down the line, stay tuned for more updates. In the meantime you can take a look at the `self-consistency example <https://github.com/normal-computing/outlines/blob/main/examples/self_consistency.py>`_.


Batching
--------

Outlines will soon allow you to vectorize model calls.
