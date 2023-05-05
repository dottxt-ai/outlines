Prompting
=========

Outlines provides a powerful domain-specific language to write and manage prompts, via what we call *prompt functions*. Prompt functions are Python functions that contain the prompt in their docstring; their arguments correspond to the variables used in the prompt. We use the Jinja library to render the prompts, with a few tweaks to make the prompt writing experience nicer.


One thus doesn't need extra abstraction to write a prompt with few-shot examples, Jinja can handle that:

.. code::

   import outlines.text as text

   @text.prompt
   def few_shots(instructions, examples, question):
       """"{{ instructions }}

       {% for examples in examples %}
       Q: {{ example.question }}
       A: {{ example.answer }}
       {% endfor %}
       Q: {{ question }}
       """

    prompt = few_shots(question, examples, question)

The original template is still accessible by calling:

.. code::

   prompt.template


Outlines also provides a few utilities to simplify workflows that connect tools to LLMs (Toolformer, ViperGPT, AutoGPT). We noticed that the same information was always repeated: once when implementing the function, the second time when writing the instructions in the prompt. No need to do this with Outlines, information can be directly pulled from the function definition:

.. code::

   import outlines.text as text

   def my_tool(arg1: str, arg2: int):
       """Tool description.

       The rest of the docstring
       """
       pass

   @text.prompt
   def tool_prompt(question, tool):
       """{{ question }}

       COMMANDS
       1. {{ tool | name }}: {{ tool | description }}, args: {{ tool | args }}

      {{ tool | source }}
       """

The same goes for output validation: the code is implemented once when defining the parser, a second time when passing the format to the prompt:

.. code::

   from pydantic import BaseModel

   import outlines.text as text

   class MyResponse(BaseModel):
       field1: int
       field2: str

   @text.prompt
   def my_prompt(response_model):
       """{{ response_model | schema }}"""

Please refer to the `Jinja documentation <https://jinja.palletsprojects.com/en/3.1.x/>`_ for more information about the syntax of the templating language.
