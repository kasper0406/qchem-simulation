import chex
import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jaxtyping import Array

from src.electronic_optimization import ElectronicAttention, SlaterDeterminant, calculate_distances


def test_permutation_equivariance_electronic_attention():
    num_electrons = 5
    num_nuclei = 2
    hdim = 8
    rngs = nnx.Rngs(0)
    eatt = ElectronicAttention(hdim, num_electrons, num_nuclei, rngs=rngs)

    # Setup the initial distances
    electron_positions = jax.random.normal(rngs.electron(), shape=(num_electrons, 3))
    nucleus_positions = jax.random.normal(rngs.nucleus(), shape=(num_nuclei, 3))

    dist = calculate_distances(electron_positions, nucleus_positions)
    original_out = eatt(dist)

    def _swap_and_check(i, j):
        # Swap the position of electron i and j
        swapped_positions = electron_positions.at[[i, j], :].set(electron_positions[[j, i], :])
        swapped_dist = calculate_distances(swapped_positions, nucleus_positions)
        swapped_out = eatt(swapped_dist)
        permuted_original = original_out.at[[j, i], :].set(original_out[[i, j], :])
        chex.assert_trees_all_close(swapped_out, permuted_original)

    _swap_and_check(0, 1)
    _swap_and_check(0, 2)
    _swap_and_check(1, 3)
    _swap_and_check(1, 4)


def test_permutation_of_slater_determinant():
    num_electrons = 5
    num_nuclei = 2
    hdim = 8
    rngs = nnx.Rngs(0)
    slater_det = SlaterDeterminant(num_electrons, num_nuclei, hdim, rngs)

    # Setup the initial distances
    electron_positions = jax.random.normal(rngs.electron(), shape=(num_electrons, 3))
    nucleus_positions = jax.random.normal(rngs.nucleus(), shape=(num_nuclei, 3))

    dist = calculate_distances(electron_positions, nucleus_positions)
    spins = (-1) ** jnp.arange(num_electrons)  # Alternating spins for testing
    original_det, original_sign = slater_det(dist, spins)

    def _swap_and_check(i, j):
        # Swap the position of electron i and j
        swapped_positions = electron_positions.at[[i, j], :].set(electron_positions[[j, i], :])
        swapped_dist = calculate_distances(swapped_positions, nucleus_positions)
        swapped_spins = spins.at[[i, j], ...].set(spins[[j, i], ...])
        swapped_det, swapped_sign = slater_det(swapped_dist, swapped_spins)
        chex.assert_trees_all_close(original_det, swapped_det, rtol=1e-4)
        chex.assert_trees_all_close(original_sign, -swapped_sign, rtol=1e-4)  # Swapping two electrons will pick up a minus sign!

    _swap_and_check(0, 1)
    _swap_and_check(0, 2)
    _swap_and_check(1, 3)
    _swap_and_check(1, 4)
