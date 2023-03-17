from outlines.graph import io_toposort
from outlines.text.var import StringConstant


def compile(symbolic_inputs, outputs):
    sorted = io_toposort(symbolic_inputs, outputs)

    def fn(*inputs):
        storage_map = {s: v for s, v in zip(symbolic_inputs, inputs)}

        for node in sorted:
            for i in node.inputs:
                if isinstance(i, StringConstant):
                    storage_map[i] = i.value
            inputs = [storage_map[i] for i in node.inputs]
            results = node.op.perform(*inputs)
            for i, o in enumerate(node.outputs):
                storage_map[o] = results[i]

        if len(outputs) == 1:
            return storage_map[outputs[0]]
        else:
            return tuple(storage_map[o] for o in outputs)

    return fn
