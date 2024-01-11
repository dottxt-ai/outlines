# Prompt templating

Outlines provides a powerful domain-specific language to write and manage
prompts, via what we call *prompt functions*.  Prompt functions are Python
functions that contain a template for the prompt in their docstring, and their
arguments correspond to the variables used in the prompt. When called, a prompt
function returns the template rendered with the values of the arguments.

The aim of prompt functions is to solve several recurrent problems with prompting:

1. **Building complex prompts quickly leads to messy code.** This problem has
   already been solved in the web development community by using templating, so
   why not use it here?
2. **Composing prompts is difficult.** Why not just compose functions?
3. **Separating prompts from code.** Encapsulation in functions allows a clean
   separation between prompts and code. Moreover, like any function, prompt
   functions can be imported from other modules.

Outlines uses the [Jinja templating
engine](https://jinja.palletsprojects.com/en/3.1.x/) to render prompts, which
allows to easily compose complex prompts.

!!! warning "Prompt rendering"

    Prompt functions are opinionated when it comes to prompt rendering. These opinions are meant to avoid common prompting errors, but can have unintended consequences if you are doing something unusual. We advise to always print the prompt before using it. You can also [read the
    reference](#formatting-conventions) section if you want to know more.

## Your first prompt

The following snippet showcases a very simple prompt. The variables between
curly brackets `{{  }}` are placeholders for the values of the arguments you
will pass to the prompt function.

=== "Code"

    ```python
    import outlines

    @outlines.prompt
    def greetings(name, question):
        """Hello, {{ name }}!
        {{ question }}
        """

    prompt = greetings("user", "How are you?")
    print(prompt)
    ```

=== "Output"

    ```text
    Hello, user!
    How are you?
    ```

If a variable is missing in the function's arguments, Jinja2 will throw an `UndefinedError` exception:

=== "Code"

    ```python
    import outlines

    @outlines.prompt
    def greetings(name):
        """Hello, {{ surname }}!"""

    prompt = greetings("user")
    ```

=== "Output"

    ```text
    Traceback (most recent call last):
      File "<stdin>", line 9, in <module>
      File "/home/remi/projects/normal/outlines/outlines/prompts.py", line 38, in __call__
          return render(self.template, **bound_arguments.arguments)
      File "/home/remi/projects/normal/outlines/outlines/prompts.py", line 213, in render
          return jinja_template.render(**values)
      File "/home/remi/micromamba/envs/outlines/lib/python3.9/site-packages/jinja2/environment.py", line 1301, in render
          self.environment.handle_exception()
      File "/home/remi/micromamba/envs/outlines/lib/python3.9/site-packages/jinja2/environment.py", line 936, in handle_exception
          raise rewrite_traceback_stack(source=source)
      File "<template>", line 1, in top-level template code
      jinja2.exceptions.UndefinedError: 'surname' is undefined
    ```

## Importing prompt functions

Prompt functions are functions, and thus can be imported from other modules:

=== "prompts.py"
    ```python
    import outlines

    @outlines.prompt
    def greetings(name, question):
        """Hello, {{ name }}!
        {{ question }}
        """
    ```

=== "generate.py"

    ```python
    from .prompts import greetings

    prompt = greetings("John Doe", "How are you today?")
    ```

=== "Output"

    ```text
    Hello, John Doe!
    How are you today?
    ```

## Few-shot prompting

Few-shot prompting can lead to messy code. Prompt functions allow you to loop
over lists or dictionaries from the template. In the following example we
demonstrate how we can generate a prompt by passing a list of dictionaries with
keys `question` and `answer` to the prompt function:

=== "Code"

    ```python
    import outlines

    @outlines.prompt
    def few_shots(instructions, examples, question):
        """{{ instructions }}

        Examples
        --------

        {% for example in examples %}
        Q: {{ example.question }}
        A: {{ example.answer }}

        {% endfor %}
        Question
        --------

        Q: {{ question }}
        A:
        """

    instructions = "Please answer the following question following the examples"
    examples = [
        {"question": "2+2=?", "answer":4},
        {"question": "3+3=?", "answer":6}
    ]
    question = "4+4 = ?"

    prompt = few_shots(instructions, examples, question)
    print(prompt)
    ```

=== "Output"

    ```text
    Please answer the following question following the examples

    Examples
    --------

    Q: 2+2=?
    A: 4

    Q: 3+3=?
    A: 6

    Question
    --------

    Q: 4+4 = ?
    A:
    ```

## Conditionals, filters, etc.

Jinja2 has many features beyond looping that are not described here:
conditionals, filtering, formatting, etc. Please refer to the [Jinja
documentation](https://jinja.palletsprojects.com/en/3.1.x/>) for more
information about the syntax of the templating language. The Jinja syntax is
powerful, and we recommend you take some time to read their documentation if you
are building complex prompts.


## Tools

Several projects (e.g.[Toolformer](https://arxiv.org/abs/2302.04761), [ViperGPT](https://viper.cs.columbia.edu/), [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT), etc.) have shown that we can "teach" language models to use external functions by describing what these functions do in the prompt. In these projects the same information is often repeated twice: the function implementation, name, docstring, or arguments are copy-pasted in the prompt. This is cumbersome and error prone; you can directly pull this information from within an Outlines prompt function:

=== "Code"

    ```python
    import outlines

    def my_tool(arg1: str, arg2: int):
        """Tool description.

        The rest of the docstring
        """
        pass

    @outlines.prompt
    def tool_prompt(question, tool):
        """{{ question }}

        COMMANDS
        1. {{ tool | name }}: {{ tool | description }}, args: {{ tool | args }}

        {{ tool | source }}
        """

    prompt = tool_prompt("Can you do something?", my_tool)
    print(prompt)
    ```

=== "Output"

    ```text
    Can you do something?

    COMMANDS
    1. my_tool: Tool description, args: arg1:str, arg2:int

    def my_tool(arg1: str, arg2: int):
        """Tool description.

        The rest of the docstring
        """
        pass
    ```

## JSON response format

To build reliable chains with language models we often need to instruct them the
format in which we would like them to return their response.

Without prompt templating, the information is repeated twice between creating
the parsing function (e.g. a Pydantic model), and writing the desired schema in
the prompt. This can lead to errors that are hard to debug.

Outlines allows you to directly pull the JSON schema of a pydantic model, or
pretty print a dictionary from within an Outlines prompt function

=== "Code"

    ```python
    from pydantic import BaseModel, Field

    import outlines

    class MyResponse(BaseModel):
        field1: int = Field(description="an int")
        field2: str

    @outlines.prompt
    def my_prompt(response_model):
        """{{ response_model | schema }}"""

    my_prompt(MyResponse)
    # {
    #   "field1": "an int",
    #   "field2": "<field2>"
    # }
    ```

=== "Output"

    ```python
    response = {
        "field1": "<field1>",
        "field2": "a string"
    }

    my_prompt(MyResponse)
    # {
    #   "field1": "<field1>",
    #   "field2": "a string"
    # }
    ```

## Formatting conventions

Prompt functions are opinionated when it comes to rendering, and these opinions
are meant to avoid prompting mistakes and help with formatting.

### Whitespaces

If you have experience working with strings between triple quotes you know that
indenting has an influence on the string's formatting. Prompt functions adopt
a few conventions so you don't have to think about indents when writing prompt.

First, whether you start the prompt right after the triple quotes or on the line
below does not matter for formatting:

=== "Code"

    ```python
    import outlines

    @outlines.prompt
    def prompt1():
        """My prompt
        """

    @outlines.prompt
    def prompt2():
        """
        My prompt
        """

    print(prompt1())
    print(prompt2())
    ```

=== "Output"

    ```text
    My prompt
    My prompt
    ```

Indentation is relative to the second line of the docstring, and leading spaces are removed:

=== "Code"

    ```python
    import outlines

    @outlines.prompt
    def example1():
        """First line
        Second line
        """

    @outlines.prompt
    def example2():
        """
          Second line
          Third line
        """

    @outlines.prompt
    def example3():
        """
          Second line
            Third line
        """

    print(example1())
    print(example2())
    print(example3())
    ```

=== "Output"

    ```text
    First line
    Second line

    Second line
    Third line

    Second line
      Third line
    ```

Trailing whitespaces are not removed, unless they follow a linebreak symbol `\` (see [linebreaks](#linebreaks)).

### Linebreaks

You can use the backslash `\` to break a long line of text. It will render as a single line:

=== "Code"

    ```python
    import outlines

    @outlines.prompt
    def example():
       """
       Break in \
       several lines \
       But respect the indentation
           on line breaks.
       And after everything \
       Goes back to normal
       """

    print(example())
    ```

=== "Output"

    ```text
    Break in several lines But respect the indentation
        on line breaks.
    And after everything Goes back to normal
    ```
