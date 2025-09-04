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


def clip_grad_elementwise(min_val, max_val):
    """
    Decorator factory to clip the gradients of a function.

    Args:
        min_val (float): The minimum value to clip the gradient to.
        max_val (float): The maximum value to clip the gradient to.

    Returns:
        A decorator that can be applied to any JAX function.
    """
    def decorator(fun):
        @jax.custom_vjp
        def wrapped_fun(*args, **kwargs):
            return fun(*args, **kwargs)

        def wrapped_fun_fwd(*args, **kwargs):
            # Unfortunately jax.linearize/linear_transpose does not work well with integer inputs
            # y, jvp = jax.linearize(fun, *args, **kwargs)
            # return y, (jvp, args, kwargs)

            y = fun(*args, **kwargs)
            return y, (args, kwargs)

        def wrapped_fun_bwd(res, g):
            # Unfortunately jax.linearize/linear_transpose does not work well with integer inputs
            # jvp, args, kwargs = res
            # vjp = jax.linear_transpose(jvp, *args, **kwargs)
            # grads = vjp(g)

            args, kwargs = res
            _, vjp_fn = jax.vjp(fun, *args, **kwargs)
            grads = vjp_fn(g)

            # Clip the gradients. jax.tree_util.tree_map handles pytrees
            # (e.g., tuples/lists of gradients for multiple arguments).
            def clipper(grad: Array):
                if grad.dtype == jax.dtypes.float0:
                    # A float0 array will not be clipped
                    return grad
                return jnp.clip(grad, min_val, max_val)
            clipped_grads = jax.tree_util.tree_map(clipper, grads)

            return clipped_grads

        wrapped_fun.defvjp(wrapped_fun_fwd, wrapped_fun_bwd)
        return wrapped_fun

    return decorator


def clip_by_global_norm(grads, max_norm: float, eps: float = 1e-9):
    def _tree_l2_sq(tree):
        def leaf_sq(x):
            if getattr(x, 'dtype', None) == jax.dtypes.float0:
                return 0.0
            return jnp.sum(jnp.square(x))
        leaves = jax.tree_util.tree_leaves(tree)
        return sum(leaf_sq(x) for x in leaves) if leaves else 0.0

    l2_sq = _tree_l2_sq(grads)
    global_norm = jnp.sqrt(l2_sq)
    scale = jnp.minimum(1.0, max_norm / (global_norm + eps))
    return jax.tree_util.tree_map(
        lambda g: g if getattr(g, 'dtype', None) == jax.dtypes.float0 else g * scale,
        grads,
    )

def clip_grad_global(max_norm: float, eps: float = 1e-9):
    def decorator(fun):
        @jax.custom_vjp
        def wrapped_fun(*args, **kwargs):
            return fun(*args, **kwargs)

        def wrapped_fun_fwd(*args, **kwargs):
            # Unfortunately jax.linearize/linear_transpose does not work well with integer inputs
            # y, jvp = jax.linearize(fun, *args, **kwargs)
            # return y, (jvp, args, kwargs)

            y = fun(*args, **kwargs)
            return y, (args, kwargs)

        def wrapped_fun_bwd(res, g):
            # Unfortunately jax.linearize/linear_transpose does not work well with integer inputs
            # jvp, args, kwargs = res
            # vjp = jax.linear_transpose(jvp, *args, **kwargs)
            # grads = vjp(g)

            args, kwargs = res
            _, vjp_fn = jax.vjp(fun, *args, **kwargs)
            grads = vjp_fn(g)

            clipped_grads = clip_by_global_norm(grads, max_norm, eps)
            return clipped_grads

        wrapped_fun.defvjp(wrapped_fun_fwd, wrapped_fun_bwd)
        return wrapped_fun

    return decorator
