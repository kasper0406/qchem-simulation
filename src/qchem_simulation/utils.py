import jax
from dataclasses import dataclass, field
from jaxtyping import Array, Key, ArrayLike

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
