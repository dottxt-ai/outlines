Prompting
=========

Outlines provides a powerful domain-specific language to write and manage prompts, via what we call *prompt functions*. Prompt functions are Python functions that contain a template for the prompt in their docstring, and their arguments correspond to the variables used in the prompt. When called, a prompt function returns the template rendered with the values of the arguments:

.. code::

   import outlines.text as text

   @text.prompt
   def greetings(name, question):
       """Hello, {{ name }}!
       {{ question }}
       """

    prompt = greetings("user", "How are you?")
    # Hello, user!
    # How are you?


Outlines uses the `Jinja templating engine <https://jinja.palletsprojects.com/en/3.1.x/>`_ to render prompts, which allows to easily compose complex prompts. No need for extra abstractions to write a prompt with few-shot examples, Jinja can handle that:

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


Please refer to the `Jinja documentation <https://jinja.palletsprojects.com/en/3.1.x/>`_ for more information about the syntax of the templating language. The Jinja syntax is powerful, and we recommend you take some time to read their documentation if building your prompts requires complex logic involving for instance loops and conditionals.


Calling tools
~~~~~~~~~~~~~

Several projects (e.g.`Toolformer <https://arxiv.org/abs/2302.04761>`_, `ViperGPT <https://viper.cs.columbia.edu/>`_, `AutoGPT <https://github.com/Significant-Gravitas/Auto-GPT>`_, etc.) have shown that we can "teach" language models to use external functions by describing what these functions do in the prompt. In these projects the same information is often repeated twice: the function implementation, name, docstring, or arguments are copy-pasted in the prompt. This is cumbersome and error prone; you can directly pull this information from within an Outlines prompt function:

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

    tool_prompt("Can you do something?", my_tool)
    # Can you do something?
    #
    # COMMANDS
    # 1. my_tool: Tool description, args: arg1:str, arg2:int
    #
    # def my_tool(arg1: str, arg2: int):
    #     """Tool description.
    #
    #     The rest of the docstring
    #     """
    #     pass


Specify a response format
~~~~~~~~~~~~~~~~~~~~~~~~~

To build reliable chains with language models we often need to instruct them the format in which we would like them to return their response. Again the information is often repeated twice between creating the parsing function, and writing the desired schema in the prompt. You can directly pull the JSON schema of a pydantic model, or pretty print a dictionary from within an Outlines prompt function

.. code::

   from pydantic import BaseModel, Field

   import outlines.text as text

   class MyResponse(BaseModel):
       field1: int = Field(description="an int")
       field2: str

   @text.prompt
   def my_prompt(response_model):
       """{{ response_model | schema }}"""

   my_prompt(MyResponse)
   # {
   #   "field1": "an int",
   #   "field2": "<field2>"
   # }


.. code::

   response = {
       "field1": "<field1>",
       "field2": "a string"
   }

   my_prompt(MyResponse)
   # {
   #   "field1": "<field1>",
   #   "field2": "a string"
   # }
