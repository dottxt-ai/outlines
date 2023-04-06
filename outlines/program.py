import itertools
import textwrap
import time
from functools import singledispatchmethod
from typing import Callable, Iterable, Reversible

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel

from outlines.graph import Variable, io_toposort
from outlines.text.models import LanguageModel
from outlines.text.var import StringConstant

COLORS = itertools.cycle(["deep_sky_blue2", "gold3", "deep_pink2"])


class Program:
    """ """

    def __init__(self, inputs: Iterable[Variable], outputs: Reversible[Variable]):
        self.inputs = inputs
        self.outputs = outputs
        self.frames = io_toposort(inputs, outputs)

        self.language_models = list(
            {
                node.op
                for node in self.frames
                if node.op is not None and isinstance(node.op, LanguageModel)
            }
        )
        self.lm_colors = {lm: next(COLORS) for lm in self.language_models}

        self.console = Console()

    def build_layout(self) -> Layout:
        """Create the layout for the command line interface.

        +-------------------------------------+
        | Logo + instructions                 |
        +-------------------------------------+
        | List of Ops   |  Executed trace     |
        | + parameters  |                     |
        +-------------------------------------+

        """
        layout = Layout(name="root")
        layout.split_column(Layout(name="header", size=12), Layout(name="execution"))
        layout["execution"].split_row(
            Layout(name="models"), Layout(name="script", ratio=4)
        )

        return layout

    def print_ops_description(self) -> Panel:
        """Create the model panel.

        The `model` panel displays each `Op` used in the program and their
        parameters in the color that was assigned to them. The color matches
        the color used in the `script` panel for the text they generate.

        """
        model_str = "\n\n".join(
            [f"[{self.lm_colors[lm]}] {lm.name} [/]" for lm in self.language_models]
        )
        return Panel(
            model_str,
            border_style="bright_black",
            title="[bright_black]Models[/]",
            title_align="left",
        )

    def print_header(self) -> str:
        """Display the program's header in the console."""

        welcome_ascii = textwrap.dedent(
            r"""
               ___        _   _ _
              / _ \ _   _| |_| (_)_ __   ___  ___
             | | | | | | | __| | | '_ \ / _ \/ __|
             | |_| | |_| | |_| | | | | |  __/\__ \
              \___/ \____|\__|_|_|_| |_|\___||___/
        """
        )

        text = f"[bold green]{welcome_ascii}[/bold green]\n\n"
        text += "[bright_black]Type Ctrl-C to interrupt the execution and return the current trace.[/]\n"

        return text

    def print_trace(
        self, script: str = "", elapsed_time_s: float = 0, words: int = 0
    ) -> Panel:
        """Display the current script."""
        subtitle_str = f"[bright_black]Words:[/] [bold red]{words}[/] | "
        subtitle_str += (
            f"[bright_black]Time Elapsed:[/][bold yellow] {elapsed_time_s:.1f}s [/]"
        )
        return Panel(
            script,
            border_style="bright_black",
            title="[bright_black]Script[/]",
            title_align="left",
            subtitle=subtitle_str,
            subtitle_align="right",
        )

    def execute_frames(self, *values):
        storage_map = {s: v for s, v in zip(self.inputs, values)}
        script_fmt = ""
        trace = {"script": "", "nodes": {}}

        start_time = time.time()
        time_elapsed_s = 0

        # Corner case where the users only passes strings
        if len(self.frames) == 0:
            trace["script"] = "".join(values)

        try:
            with Live(self.layout, console=self.console) as live:
                self.layout["script"].update(self.print_trace())
                live.update(self.layout)

                for node in self.frames:
                    input_fmt = self.process_frame_inputs(node, storage_map)
                    script_fmt += input_fmt
                    self.layout["script"].update(
                        self.print_trace(
                            script_fmt, time_elapsed_s, len(script_fmt.split())
                        )
                    )
                    live.update(self.layout)

                    self.execute_frame(node, storage_map, trace)
                    time_elapsed_s = time.time() - start_time

                    output_fmt = self.process_frame_outputs(node, storage_map)
                    script_fmt += output_fmt
                    self.layout["script"].update(
                        self.print_trace(
                            script_fmt, time_elapsed_s, len(script_fmt.split())
                        )
                    )
                    live.update(self.layout)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            raise e
        finally:
            decoded_script = tuple(storage_map[output] for output in self.outputs)
            trace["script"] = decoded_script
            if len(decoded_script) == 1:
                trace["script"] = decoded_script[0]
                return trace
            return trace

    def debug(self, *values):
        storage_map = {s: v for s, v in zip(self.inputs, values)}
        trace = {"script": "", "nodes": {}}
        for node in self.frames:
            self.process_frame_inputs(node, storage_map)
            self.execute_frame(node, storage_map, trace)

        return storage_map

    def process_frame_inputs(self, node, storage_map):
        """Process the nodes' inputs.

        If either of the node's inputs is a `StringConstant` we add its
        value to the storage map and return its (formatted) value to
        be added to the current value of the decoded script.

        """
        input_str, input_fmt = "", ""
        for var in node.inputs:
            if isinstance(var, StringConstant):
                if var not in storage_map:
                    storage_map[var] = var.value
                    input_str = var.value
                    input_fmt = self.format_display(None, input_str)

        return input_fmt

    def execute_frame(self, node, storage_map, trace):
        """Execute the current frame."""
        node_inputs = [storage_map[i] for i in node.inputs]
        results = node.op.perform(*node_inputs)
        for i, o in enumerate(node.outputs):
            storage_map[o] = results[i]
            trace[o.name] = results[i]

    def process_frame_outputs(self, node, storage_map):
        """Process the node's outputs.

        If the node's `Op` is a `LanguageModel` we append its
        result to the current value of the decoded script.

        """
        output_str, output_fmt = "", ""
        if isinstance(node.op, LanguageModel):
            output_str = storage_map[node.outputs[0]]
            output_fmt = self.format_display(node.op, output_str)

        return output_fmt

    @singledispatchmethod
    def format_display(self, op, text):
        return f"[white]{text}[/]"

    @format_display.register(LanguageModel)
    def format_display_LanguageModel(self, op, text):
        return f"[{self.lm_colors[op]}]{text}[/]"

    def run(self, *values):
        self.layout = self.build_layout()
        self.layout["header"].update(self.print_header())
        self.layout["models"].update(self.print_ops_description())
        return self.execute_frames(*values)


program = Program


def chain(input_vars, output_vars) -> Callable:
    """Return a function that will compute the outputs of a chain from its outputs.

    Parameters
    ----------
    input_vars
        Sequence of symbolic variables that correspond to the function's
        parameters.
    output_vars
        Symbolic variable(s) representing the expression(s) to compute.

    """

    if not isinstance(input_vars, (list, tuple)):
        raise Exception(
            "Input variables of the `compile` function should be contained in a list or a tupe, even when there is a single input."
        )
    if not isinstance(output_vars, (list, tuple)):
        output_vars = (output_vars,)

    sorted_nodes = io_toposort(input_vars, output_vars)

    def function(*inputs):
        storage_map = {s: v for s, v in zip(input_vars, inputs)}

        for node in sorted_nodes:
            for i in node.inputs:
                if isinstance(i, StringConstant):
                    storage_map[i] = i.value
            inputs = [storage_map[i] for i in node.inputs]
            results = node.op.perform(*inputs)
            for i, o in enumerate(node.outputs):
                storage_map[o] = results[i]

        if len(output_vars) == 1:
            return storage_map[output_vars[0]]
        else:
            return tuple(storage_map[o] for o in output_vars)

    return function
