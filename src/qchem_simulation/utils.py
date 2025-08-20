import jax
from dataclasses import dataclass, field
from jaxtyping import Array, Key, ArrayLike
import jax.numpy as jnp

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
