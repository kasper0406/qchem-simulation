import jax
from dataclasses import dataclass, field
from jaxtyping import Array, Key, ArrayLike
import jax.numpy as jnp
from typing import Callable
from jaxtyping import Array
from flax import nnx
import einops


def reduce_vmap(
    *,
    reduce_fn: Callable,
    acc_fn: Callable,
    batch_size: int,
    init: Array,
    in_axes: int | None | tuple[int] = 0,
    out_axes: int | None | tuple[int] = 0,
):
    if out_axes != 0:
        # To relax this, the scan should carry all out_axes
        raise ValueError("Only out_axes=0 is supported currently.")

    def decorator(f: Callable):
        def _internal(*args):
            # TODO(knielsen): Handle the case where in_axes is int | None
            def reshape_arg(arg, batch_axis):
                if batch_axis is None:
                    return arg
            
                def reshape_leaf(leaf):
                    # Put the batch axis as axis 0, and reshape along batch_axis dimension
                    leaf_swapped = jnp.moveaxis(leaf, batch_axis, 0)
                    new_shape = (-1, batch_size) + leaf_swapped.shape[1:]
                    return leaf_swapped.reshape(new_shape)

                return jax.tree.map(reshape_leaf, arg)

            args = [reshape_arg(arg, batch_axis) for arg, batch_axis in zip(args, in_axes)]

            # We have reshaped the input such that the batches axis is always axis 0
            updated_inaxes = tuple(0 if axis is not None else None for axis in in_axes)
            @nnx.scan(in_axes=(nnx.Carry, *updated_inaxes), out_axes=(nnx.Carry))
            def scan_batch(acc, *args):
                mini_batch = nnx.vmap(f, in_axes=updated_inaxes, out_axes=out_axes)(*args)
                # TODO(knielsen): This also assumes out_axes=0
                return acc_fn(acc, reduce_fn(mini_batch, axis=0))

            result = scan_batch(init, *args)
            return result

        return _internal

    return decorator


@jax.tree_util.register_dataclass
@dataclass
class Electron:
    position: Array
    spin: Array

@jax.tree_util.register_dataclass
@dataclass
class Nucleus:
    position: Array
    charge: Array


def clip_grad(min_val, max_val):
    """
    Decorator factory to clip the gradients of a function.

    Args:
        min_val (float): The minimum value to clip the gradient to.
        max_val (float): The maximum value to clip the gradient to.

    Returns:
        A decorator that can be applied to any JAX function.
    """
    # This is the decorator that will be returned
    def decorator(fun):
        # Create a function with a custom VJP
        @jax.custom_vjp
        def wrapped_fun(*args, **kwargs):
            # The forward pass is just the original function
            return fun(*args, **kwargs)

        # Define the forward pass for the custom VJP
        def wrapped_fun_fwd(*args, **kwargs):
            # We need the original inputs for the backward pass to re-compute the VJP
            # The output is (results, residuals_for_backward_pass)
            return fun(*args, **kwargs), (args, kwargs)

        # Define the custom backward pass
        def wrapped_fun_bwd(res, g):
            args, kwargs = res  # Unpack the inputs from the forward pass

            # Get the VJP of the original function
            # jax.vjp returns (primals_out, vjp_fn)
            _, vjp_fn = jax.vjp(fun, *args, **kwargs)

            # Compute the original gradients
            original_grads = vjp_fn(g)

            # Clip the gradients. jax.tree_util.tree_map handles pytrees
            # (e.g., tuples/lists of gradients for multiple arguments).
            clipper = lambda grad: jnp.clip(grad, min_val, max_val)
            clipped_grads = jax.tree_util.tree_map(clipper, original_grads)

            return clipped_grads

        # Set the custom VJP for our wrapped function
        wrapped_fun.defvjp(wrapped_fun_fwd, wrapped_fun_bwd)

        return wrapped_fun

    return decorator
