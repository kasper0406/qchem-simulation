"""Born–Oppenheimer Coulomb and electronic kinetic energies in atomic units."""

from dataclasses import dataclass
from functools import partial
from typing import Literal
import warnings

import chex
from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from .utils import Electron, Nucleus
from .wavefunction import WaveFunction


@jax.tree_util.register_dataclass
@dataclass
class HamiltonianParts:
    kinetic_energy: Array
    potential_energy: Array

    # Partial contributions to the potential energy
    electron_repulsion: Array
    nucleus_electron_interaction: Array
    nucleus_nucleus_interaction: Array


def calculate_kinetic_energy_per_electron(wave_function: "WaveFunction", electrons: Electron, nuclei: Nucleus) -> Array:
    """
    Compute the local kinetic energy -0.5 * (∇^2 psi) / psi by vmapping
    `folx.forward_laplacian` over each electron's 3D coordinates individually.

    Assumes wave_function returns the log-amplitude log(psi) of the wave function (returns both amplitude and sign).
    """
    from folx import forward_laplacian
    position_indices = jnp.arange(electrons.position.shape[0])

    @nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
    def calc_terms(wave_function: WaveFunction, position: Array, electron_idx: int) -> Array:
        # Evaluate the wave function with respect to the electron's position
        def calc_wrt_position(position: Array, graphdef: nnx.GraphDef, state: nnx.State) -> Array:
            wave_function = nnx.merge(graphdef, state)
            log_amplitude, _sign, _ortho_measure = wave_function.eval_wrt_electron_position(position, nuclei, electron_idx, electrons.position, electrons.spin)
            return log_amplitude

        graphdef, state = nnx.split(wave_function)
        laplacian = forward_laplacian(partial(calc_wrt_position, graphdef=graphdef, state=state))(position)
        # print(f"Jacobian: {amplitude_laplacian.jacobian}")
        # print(f"Laplacian: {amplitude_laplacian}")

        # Use the identity: ∇^2 psi = psi * (∇^2 log(psi) + |∇log(psi)|^2)
        # We actually want to calculate (∇^2 psi)/psi (the local energy, and from linearity of the Laplacian term
        # in the kinetic energy). Therefore, we calculate: (∇^2 psi)/psi = (∇^2 log(psi) + |∇log(psi)|^2)
        # where |∇log(psi)|^2 = J \dot J^T where J is the Jacobian of the wave function
        # Notice that since the laplacian is linear, the sign of psi will cancel out in (∇^2 psi)/psi.
        chex.assert_shape(laplacian.dense_jacobian, (3, ))
        jacobian_squared = jnp.sum(laplacian.dense_jacobian * laplacian.dense_jacobian, axis=-1)
        local_kinetic_term = laplacian.laplacian + jacobian_squared
        return local_kinetic_term

    terms = calc_terms(wave_function, electrons.position, position_indices)
    # print(f"Laplacian terms shape: {terms.shape}")
    return -0.5 * jnp.sum(terms)


def calculate_kinetic_energy_full(
    wave_function: "WaveFunction",
    electrons: Electron,
    nuclei: Nucleus,
    sparsity_threshold: int | float = 0,
) -> Array:
    """
    Compute the local kinetic energy -0.5 * (∇^2 psi) / psi in a single
    `folx.forward_laplacian` pass over all 3N electron coordinates.

    Assumes wave_function returns the log-amplitude log(psi) of the wave function (returns both amplitude and sign).
    If Folx fails in the sparse Hessian contraction, retry with the exact dense path.
    """
    from folx import forward_laplacian

    num_electrons = electrons.position.shape[0]
    flat_positions = electrons.position.reshape(-1)
    graphdef, state = nnx.split(wave_function)

    def calc_wrt_positions(flat_pos: Array) -> Array:
        wf = nnx.merge(graphdef, state)
        elec = Electron(position=flat_pos.reshape(num_electrons, 3), spin=electrons.spin)
        log_amplitude, _sign, _ortho_measure = wf(elec, nuclei)
        return log_amplitude

    try:
        laplacian = forward_laplacian(
            calc_wrt_positions, sparsity_threshold=sparsity_threshold
        )(flat_positions)
    except IndexError as error:
        # Folx 0.2.31 can index past the sparse vmap axes in an attention dot product.
        # Keep unrelated indexing errors visible to the caller.
        frame = error.__traceback__
        while frame.tb_next is not None:
            frame = frame.tb_next
        if (
            not sparsity_threshold
            or frame.tb_frame.f_globals.get("__name__") != "folx.hessian"
            or frame.tb_frame.f_code.co_name != "vmapped_jac_hessian_jac"
        ):
            raise
        warnings.warn(
            "Folx could not apply the sparse Hessian contraction. Using the dense forward Laplacian.",
            RuntimeWarning,
            stacklevel=2,
        )
        laplacian = forward_laplacian(calc_wrt_positions)(flat_positions)

    # Use the identity: (∇^2 psi)/psi = ∇^2 log|psi| + |∇ log|psi||^2
    chex.assert_shape(laplacian.dense_jacobian, (3 * num_electrons,))
    jacobian_squared = jnp.sum(laplacian.dense_jacobian ** 2)
    return -0.5 * (laplacian.laplacian + jacobian_squared)


def calculate_kinetic_energy(
    wave_function: "WaveFunction",
    electrons: Electron,
    nuclei: Nucleus,
    *,
    method: Literal["full", "per_electron"] = "full",
    sparsity_threshold: int | float = 0,
) -> Array:
    """
    Compute the local kinetic energy -0.5 * (∇^2 psi) / psi.

    Defaults to the full 3N forward Laplacian (`calculate_kinetic_energy_full`).
    Pass `method="per_electron"` to use `calculate_kinetic_energy_per_electron`.
    """
    if method == "full":
        return calculate_kinetic_energy_full(
            wave_function, electrons, nuclei, sparsity_threshold=sparsity_threshold
        )
    if method == "per_electron":
        return calculate_kinetic_energy_per_electron(wave_function, electrons, nuclei)
    raise ValueError(
        f"Unknown kinetic energy method: {method!r}. Expected 'full' or 'per_electron'."
    )


def _inverse_distance(displacement: Array) -> Array:
    """Use 1/r away from contact; at contact use the finite 1e12 convention.

    The exact collision has no defined Coulomb force. A constant branch gives
    that single configuration a zero displacement gradient.
    """
    squared_distance = jnp.sum(jnp.square(displacement), axis=-1)
    coincident = squared_distance == 0
    safe_squared_distance = jnp.where(coincident, 1.0, squared_distance)
    return jnp.where(coincident, 1e12, jax.lax.rsqrt(safe_squared_distance))


def nucleus_nucleus_repulsion(nuclei: Nucleus) -> Array:
    nucleus_i, nucleus_j = jnp.triu_indices(nuclei.position.shape[0], k=1)
    charges = nuclei.charge[nucleus_i] * nuclei.charge[nucleus_j]
    displacement = nuclei.position[nucleus_i] - nuclei.position[nucleus_j]
    return jnp.sum(charges * _inverse_distance(displacement))


def electron_nucleus_attraction(electrons: Electron, nuclei: Nucleus) -> Array:
    displacement = electrons.position[:, None] - nuclei.position[None, :]
    return -jnp.sum(nuclei.charge[None, :] * _inverse_distance(displacement))


def hamiltonian(wave_function: "WaveFunction", electrons: Electron, nuclei: Nucleus, key: Key):
    """Compute the local energy Hψ/ψ and its components."""
    nucleus_repulsion = nucleus_nucleus_repulsion(nuclei)

    electron_repulsion = 0.0
    if electrons.position.shape[0] > 1:
        electron_i, electron_j = jnp.triu_indices(electrons.position.shape[0], k=1)
        displacement = electrons.position[electron_i] - electrons.position[electron_j]
        electron_repulsion = jnp.sum(_inverse_distance(displacement))

    nucleus_electron_interaction = electron_nucleus_attraction(electrons, nuclei)

    kinetic_energy = calculate_kinetic_energy(wave_function, electrons, nuclei)
    potential_energy = electron_repulsion + nucleus_electron_interaction + nucleus_repulsion

    result = kinetic_energy + potential_energy
    return result, HamiltonianParts(
        kinetic_energy=kinetic_energy,
        potential_energy=potential_energy,

        electron_repulsion=electron_repulsion,
        nucleus_electron_interaction=nucleus_electron_interaction,
        nucleus_nucleus_interaction=nucleus_repulsion,
    )
