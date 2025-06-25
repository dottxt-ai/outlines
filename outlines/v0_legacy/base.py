import asyncio
import builtins
import functools
import inspect
from typing import Callable, Optional


# Allow nested loops for running in notebook. We don't enable it globally as it
# may interfere with other libraries that use asyncio.
if hasattr(builtins, "__IPYTHON__"): # pragma: no cover
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        print(
            "Couldn't patch nest_asyncio because it's not installed. Running in the notebook might be have issues"
        )


class vectorize:
    """Returns an object that acts like a function but takes arrays as an input.

    The vectorized function evaluates `func` over successive tuples of the input
    chararrays and returns a single NumPy chararrays or a tuple of NumPy chararrays.

    Its behavior is similar to NumPy's `vectorize` for Python functions: the function
    being vectorized is executed in a `for` loop. Coroutines, however, are executed
    concurrently.

    Part of the code was adapted from `numpy.lib.function_base`.

    """

    def __init__(self, func: Callable, signature: Optional[str] = None):
        self.setup_numpy(signature)
        self.func = func
        self.signature = signature
        self.is_coroutine_fn = inspect.iscoroutinefunction(func)

        functools.update_wrapper(self, func)

    def setup_numpy(self, signature: Optional[str] = None):
        """Setup NumPy and import required functions based on NumPy version.

        As numpy is an optional dependency only needed for the legacy OpenAI
        model, we put the import in a try block to make sure no error is raised
        if it's not installed.
        """
        try:
            import numpy as np
            self.np = np
            # Import required functions based on NumPy version
            np_major_version = int(np.__version__.split(".")[0])
            if np_major_version >= 2:
                from numpy.lib._function_base_impl import (
                    _calculate_shapes,
                    _parse_gufunc_signature,
                    _parse_input_dimensions,
                    _update_dim_sizes,
                )
            else: # pragma: no cover
                from numpy.lib.function_base import (
                    _calculate_shapes,
                    _parse_gufunc_signature,
                    _parse_input_dimensions,
                    _update_dim_sizes,
                )
            self._calculate_shapes = _calculate_shapes
            self._parse_gufunc_signature = _parse_gufunc_signature
            self._parse_input_dimensions = _parse_input_dimensions
            self._update_dim_sizes = _update_dim_sizes
            if signature is not None:
                # Parse the signature string into a Python data structure.
                # For instance "(m),(s)->(s,m)" becomes `([(m,),(s,)],[(s,m)])`.
                self._in_and_out_core_dimensions = self._parse_gufunc_signature(signature)
            else:
                self._in_and_out_core_dimensions = None
        except (ModuleNotFoundError, ImportError): # pragma: no cover
            self.np = None
            self._calculate_shapes = None
            self._parse_gufunc_signature = None
            self._parse_input_dimensions = None
            self._update_dim_sizes = None
            self._in_and_out_core_dimensions = None

    def __call__(self, *args, **kwargs):
        """Call the vectorized function."""
        # Raise an error ourselves if NumPy is not installed.
        if not self.np: # pragma: no cover
            raise ImportError(
                "NumPy is required to use the legacy version of the OpenAI "
                "model."
            )
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
        args = [self.np.array(arg) for arg in args]
        kwargs = {key: self.np.array(value) for key, value in kwargs.items()}

        # Broadcast args and kwargs
        broadcast_shape = self.np.broadcast(*args, *list(kwargs.values())).shape
        args = [self.np.broadcast_to(arg, broadcast_shape) for arg in args]
        kwargs = {
            key: self.np.broadcast_to(value, broadcast_shape)
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
            [self.np.asarray(x).reshape(broadcast_shape).squeeze() for x in zip(*outputs)]
        )
        outputs = tuple([x.item() if self.np.ndim(x) == 0 else x for x in outputs])

        n_results = len(list(outputs))

        return outputs[0] if n_results == 1 else outputs

    def call_with_signature(self, *args, **kwargs):
        """Call functions and coroutines when a signature is specified."""
        input_core_dims, output_core_dims = self._in_and_out_core_dimensions

        # Make sure that the numbers of arguments passed is compatible with
        # the signature.
        num_args = len(args) + len(kwargs)
        if num_args != len(input_core_dims):
            raise TypeError(
                "wrong number of positional arguments: "
                "expected %r, got %r" % (len(input_core_dims), len(args))
            )

        # Convert args and kwargs to arrays
        args = [self.np.asarray(arg) for arg in args]
        kwargs = {key: self.np.array(value) for key, value in kwargs.items()}

        # Find the arguments' broadcast shape, and map placeholder
        # variables in the signature to the number of dimensions
        # they correspond to given the arguments.
        broadcast_shape, dim_sizes = self._parse_input_dimensions(
            args + list(kwargs.values()), input_core_dims
        )

        # Calculate the shape to which each of the arguments should be broadcasted
        # and reshape them accordingly.
        input_shapes = self._calculate_shapes(broadcast_shape, dim_sizes, input_core_dims)
        args = [
            self.np.broadcast_to(arg, shape, subok=True)
            for arg, shape in zip(args, input_shapes)
        ]
        kwargs = {
            key: self.np.broadcast_to(value, broadcast_shape)
            for key, value in kwargs.items()
        }

        n_out = len(output_core_dims)

        if self.is_coroutine_fn:
            outputs = self.vectorize_call_coroutine(broadcast_shape, args, kwargs)
        else:
            outputs = self.vectorize_call(broadcast_shape, args, kwargs)

        outputs = [
            results if isinstance(results, tuple) else (results,) for results in outputs
        ]

        flat_outputs = list(zip(*outputs))
        n_results = len(flat_outputs)

        if n_out != n_results:
            raise ValueError(
                f"wrong number of outputs from the function, expected {n_out}, got {n_results}"
            )

        # The number of dimensions of the outputs are not necessarily known in
        # advance. The following iterates over the results and updates the
        # number of dimensions of the outputs accordingly.
        for results, core_dims in zip(flat_outputs, output_core_dims):
            for result in results:
                self._update_dim_sizes(dim_sizes, result, core_dims)

        # Calculate the shape to which each of the outputs should be broadcasted
        # and reshape them.
        shapes = self._calculate_shapes(broadcast_shape, dim_sizes, output_core_dims)
        outputs = tuple(
            [
                self.np.hstack(results).reshape(shape).squeeze()
                for shape, results in zip(shapes, zip(*outputs))
            ]
        )
        outputs = tuple([x.item() if self.np.ndim(x) == 0 else x for x in outputs])

        return outputs[0] if n_results == 1 else outputs

    def vectorize_call(self, broadcast_shape, args, kwargs):
        """Run the function in a for loop.

        A possible extension would be to parallelize the calls.

        Parameters
        ----------
        broadcast_shape
            The brodcast shape of the input arrays.
        args
            The function's broadcasted arguments.
        kwargs
            The function's broadcasted keyword arguments.

        """
        outputs = []
        for index in self.np.ndindex(*broadcast_shape):
            current_args = tuple(arg[index] for arg in args)
            current_kwargs = {key: value[index] for key, value in kwargs.items()}
            outputs.append(self.func(*current_args, **current_kwargs))

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
            for index in self.np.ndindex(*broadcast_shape):
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
