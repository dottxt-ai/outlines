import asyncio
import functools
import inspect

import numpy as np


class vectorize:
    """Returns an object that acts like a function but takes arrays as an input.

    The vectorized function evaluates `func` over successive tuples of the input
    chararrays and returns a single NumPy chararrays or a tuple of NumPy chararrays.

    Its behavior is similar to NumPy's `vectorize` for Python functions: the function
    being vectorized is executed in a `for` loop. Coroutines, however, are executed
    concurrently.

    Part of the code was adapted from `numpy.lib.function_base`.

    """

    def __init__(self, func, signature=None):
        self.func = func
        self.signature = signature
        self.is_coroutine_fn = inspect.iscoroutinefunction(func)

        functools.update_wrapper(self, func)

        if self.signature is not None:
            raise NotImplementedError(
                "Vectorization of non-scalar functions is not implemented yet."
            )

    def __call__(self, *args, **kwargs):
        """Call the vectorized function."""
        if not args and not kwargs:
            return self.call_thunk()
        elif self.signature is not None:
            return self.call_with_signature(*args, **kwargs)
        else:
            return self.call_no_signature(*args, **kwargs)

    def call_thunk(self):
        """Call a vectorized thunk.

        Thunks have no arguments and can thus be called directly.

        """
        if self.is_coroutine_fn:
            loop = asyncio.new_event_loop()
            try:
                outputs = loop.run_until_complete(self.func())
            finally:
                loop.close()
        else:
            outputs = self.func()

        return outputs

    def call_no_signature(self, *args, **kwargs):
        """Call functions and coroutines when no signature is specified.

        When no signature is specified we assume that all of the function's
        inputs and outputs are scalars (core dimension of zero). We first
        broadcast the input arrays, then iteratively apply the function over the
        elements of the broadcasted arrays and finally reshape the results to
        match the input shape.

        Functions are executed in a for loop, coroutines are executed
        concurrently.

        """
        # Convert args and kwargs to arrays
        args = [np.array(arg) for arg in args]
        kwargs = {key: np.array(value) for key, value in kwargs.items()}

        # Broadcast args and kwargs
        broadcast_shape = np.broadcast(*args, *list(kwargs.values())).shape
        args = [np.broadcast_to(arg, broadcast_shape) for arg in args]
        kwargs = {
            key: np.broadcast_to(value, broadcast_shape)
            for key, value in kwargs.items()
        }

        # Execute functions in a loop, and coroutines concurrently
        if self.is_coroutine_fn:
            outputs = self.vectorize_call_coroutine(broadcast_shape, args, kwargs)
        else:
            outputs = self.vectorize_call(broadcast_shape, args, kwargs)

        # `outputs` is a flat array or a tuple of flat arrays. We reshape the arrays
        # to match the input shape.
        outputs = [
            results if isinstance(results, tuple) else (results,) for results in outputs
        ]
        outputs = tuple(
            [np.asarray(x).reshape(broadcast_shape).squeeze() for x in zip(*outputs)]
        )
        outputs = tuple([x.item() if np.ndim(x) == 0 else x for x in outputs])

        n_results = len(list(outputs))

        return outputs[0] if n_results == 1 else outputs

    def vectorize_call(self, broadcast_shape, flat_args, flat_kwargs):
        """Run the function in a for loop.

        A possible extension would be to parallelize the calls.

        Parameters
        ----------
        broadcast_shape
            The brodcast shape of the input arrays.
        flat_args
            A flat array that contains the function's arguments.
        flat_kwargs
            A flat array that contains the function's keyword arguments.

        """
        outputs = []
        for index in np.ndindex(*broadcast_shape):
            args = tuple(arg[index] for arg in flat_args)
            kwargs = {key: value[index] for key, value in flat_kwargs.items()}
            outputs.append(self.func(*args, **kwargs))

        return outputs

    def vectorize_call_coroutine(self, broadcast_shape, args, kwargs):
        """Run coroutines concurrently.

        Creates as many tasks as needed and executes them in a new event
        loop.

        Parameters
        ----------
        broadcast_shape
            The brodcast shape of the input arrays.
        args
            The function's broadcasted arguments.
        kwargs
            The function's broadcasted keyword arguments.

        """

        async def create_and_gather_tasks():
            tasks = []
            for index in np.ndindex(*broadcast_shape):
                current_args = tuple(arg[index] for arg in args)
                current_kwargs = {key: value[index] for key, value in kwargs.items()}
                tasks.append(self.func(*current_args, **current_kwargs))

            outputs = await asyncio.gather(*tasks)

            return outputs

        loop = asyncio.new_event_loop()
        try:
            outputs = loop.run_until_complete(create_and_gather_tasks())
        finally:
            loop.close()

        return outputs
