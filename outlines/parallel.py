import asyncio
from typing import Awaitable, Callable, Union


def elemwise(fn: Union[Callable, Awaitable]) -> Callable:
    """Generalizes a function on arguments to lists of arguments.

    All inputs must be single elements or lists of the same size. If at least one
    input is a list and all others are a simple elements, the elements are
    "broadcasted" to lists that repeat the same element.

    For convenience, if the function being mapped is a coroutine and the inputs
    are scalar then the function is executed in a local loop.

    `outlines.elemwise` accepts both functions and coroutines, with the
    difference that it will run coroutines in parallel.

    Parameters
    ----------
    fn
        The function to generalize.

    """

    def run(*args, **kwargs):
        """Map the decorated function over lists of arguments."""
        args, kwargs, num_elements = prepare_elemwise_inputs(*args, **kwargs)

        if asyncio.iscoroutinefunction(fn):
            return async_elemwise(num_elements, *args, **kwargs)
        elif callable(fn):
            return sync_elemwise(num_elements, *args, **kwargs)
        else:
            raise TypeError(
                "The `outlines.elemwise` only works on callables and coroutines."
            )

    async def async_elemwise(num_elements, *args, **kwargs):
        """Execute async functions concurrently."""

        if num_elements == 0:
            return await fn(*args, **kwargs)

        tasks = []
        for i in range(num_elements):
            current_args = [input[i] for input in args]
            current_kwargs = {k: v[i] for k, v in kwargs.items()}
            tasks.append(asyncio.create_task(fn(*current_args, **current_kwargs)))

        return await asyncio.gather(*tasks)

    def sync_elemwise(num_elements, *args, **kwargs):
        """Execute sync functions sequentially in a `for` loop.

        # TODO: Use a pool of processes to accelerate the execution.

        """

        if num_elements == 0:
            return fn(*args, **kwargs)

        results = []
        for i in range(num_elements):
            current_args = [input[i] for input in args]
            current_kwargs = {k: v[i] for k, v in kwargs.items()}
            results.append(fn(*current_args, **current_kwargs))

        return results

    return run


def prepare_elemwise_inputs(*args, **kwargs):
    """Make sure lists are all the same size and "broadcast" other inputs."""

    values = args + tuple(kwargs.values())

    # Make sure that all lists are of the same length
    lengths = set()
    for x in values:
        if isinstance(x, list):
            lengths.add(len(x))

    if len(lengths) == 0:
        return args, kwargs, 0

    if len(lengths) > 1:
        raise TypeError(
            "All lists passed to an element-wise function must have the same length."
        )

    num_elements = next(iter(lengths))

    # Expand the individual strings (or anything that's printable)
    prepared_args = []
    for x in args:
        if isinstance(x, list):
            prepared_args.append(x)
        else:
            prepared_args.append([x] * num_elements)

    prepared_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, list):
            prepared_kwargs[k] = v
        else:
            prepared_kwargs[k] = [v] * num_elements

    return prepared_args, prepared_kwargs, num_elements
