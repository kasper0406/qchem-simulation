import chex
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qchem_simulation.electronic_optimization import (
    CuspElectronNuclei,
    CuspElectrons,
    ExponentialEnvelope,
    WaveFunction,
    calculate_distances,
    calculate_kinetic_energy,
)
from qchem_simulation.utils import Electron, Nucleus


def test_wavefunction_full_antisymmetry():
    """Swapping any two electrons (same or opposite spin) preserves log|psi| and flips sign."""
    num_electrons = 5
    num_nuclei = 2
    rngs = nnx.Rngs(42)

    wf = WaveFunction(num_electrons=num_electrons, num_nuclei=num_nuclei, rngs=rngs)

    # Set distinct non-trivial parameter values on cusps and envelope to catch any symmetry breaking.
    wf.jastrow_factor.electrons_cusp.decay_strength_parallel[...] = jnp.array(0.7)
    wf.jastrow_factor.electrons_cusp.decay_strength_antiparallel[...] = jnp.array(2.3)
    wf.jastrow_factor.nuclei_cusp.decay_strengths[...] = jnp.array([0.4, 1.9])
    wf.envelope.pi[...] = jnp.array([0.5, 1.8])
    wf.envelope.sigma[...] = jnp.array([0.8, 1.4])

    electron_positions = jax.random.normal(rngs.electron(), shape=(num_electrons, 3))
    spins = jnp.reshape((-1.0) ** jnp.arange(num_electrons), (num_electrons, 1))
    nuclei = Nucleus(
        position=jax.random.normal(rngs.nucleus(), shape=(num_nuclei, 3)),
        charge=jnp.array([2.0, 3.0]),
    )

    electrons = Electron(position=electron_positions, spin=spins)
    original_log_psi, original_sign, _ = wf(electrons, nuclei)
    assert jnp.isfinite(original_log_psi)
    assert jnp.abs(original_sign) == 1.0

    # Test both opposite-spin swaps (e.g., (0, 1), (1, 4)) and same-spin swaps (e.g., (0, 2), (1, 3))
    for i, j in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 4)]:
        swapped_positions = electron_positions.at[[i, j], :].set(electron_positions[[j, i], :])
        swapped_spins = spins.at[[i, j], :].set(spins[[j, i], :])
        swapped_electrons = Electron(position=swapped_positions, spin=swapped_spins)

        swapped_log_psi, swapped_sign, _ = wf(swapped_electrons, nuclei)
        chex.assert_trees_all_close(swapped_log_psi, original_log_psi, rtol=1e-4, atol=1e-4)
        chex.assert_trees_all_close(swapped_sign, -original_sign, rtol=1e-4)


def test_cusp_electrons_permutation_invariance_and_positivity():
    """CuspElectrons is permutation-invariant and stays finite for negative raw parameters."""
    num_electrons = 4
    cusp = CuspElectrons(num_electrons)

    # Set distinct parallel and anti-parallel raw decay strengths
    cusp.decay_strength_parallel[...] = jnp.array(0.3)
    cusp.decay_strength_antiparallel[...] = jnp.array(2.1)

    key = jax.random.PRNGKey(0)
    positions = jax.random.normal(key, shape=(num_electrons, 3))
    spins = jnp.array([[1.0], [-1.0], [1.0], [-1.0]])
    dists = calculate_distances(positions, jnp.zeros((1, 3))).electron_distances.magnitude

    original = cusp(dists, spins)
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        swapped_pos = positions.at[[i, j], :].set(positions[[j, i], :])
        swapped_spins = spins.at[[i, j], :].set(spins[[j, i], :])
        swapped_dists = calculate_distances(swapped_pos, jnp.zeros((1, 3))).electron_distances.magnitude
        chex.assert_trees_all_close(cusp(swapped_dists, swapped_spins), original, rtol=1e-5, atol=1e-6)

    # Test negative raw parameters, including a distance d=0.2 where 1 + (-5.0) * d == 0 if unconstrained
    pair_dists = jnp.full((num_electrons, num_electrons), 0.2) * (1.0 - jnp.eye(num_electrons))
    for neg_val in [-5.0, -20.0, -100.0]:
        cusp.decay_strength_parallel[...] = jnp.array(neg_val)
        cusp.decay_strength_antiparallel[...] = jnp.array(neg_val)
        val = cusp(pair_dists, spins)
        assert jnp.isfinite(val), f"Expected finite cusp value for raw param {neg_val}, got {val}"


def test_cusp_electrons_single_electron():
    """CuspElectrons with 1 electron returns 0.0."""
    cusp = CuspElectrons(num_electrons=1)
    dists = jnp.zeros((1, 1))
    spins = jnp.array([[1.0]])
    assert cusp(dists, spins) == 0.0


def test_cusp_electron_nuclei_permutation_invariance_and_positivity():
    """CuspElectronNuclei has per-nucleus params, is permutation-invariant, and stays finite for negative raw params."""
    num_electrons = 4
    num_nuclei = 3
    cusp = CuspElectronNuclei(num_electrons=num_electrons, num_nuclei=num_nuclei)

    assert cusp.decay_strengths[...].shape == (num_nuclei,)
    cusp.decay_strengths[...] = jnp.array([0.2, 1.5, 3.0])

    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    e_pos = jax.random.normal(k1, shape=(num_electrons, 3))
    n_pos = jax.random.normal(k2, shape=(num_nuclei, 3))
    charges = jnp.array([1.0, 2.0, 6.0])

    en_dists = calculate_distances(e_pos, n_pos).electron_nuclei_distances.magnitude
    original = cusp(en_dists, charges)

    for i, j in [(0, 1), (0, 3), (1, 2)]:
        swapped_e_pos = e_pos.at[[i, j], :].set(e_pos[[j, i], :])
        swapped_dists = calculate_distances(swapped_e_pos, n_pos).electron_nuclei_distances.magnitude
        chex.assert_trees_all_close(cusp(swapped_dists, charges), original, rtol=1e-5, atol=1e-6)

    # Test negative raw parameters at distance d=0.2 where 1 + (-5.0) * d == 0 if unconstrained
    fixed_dists = jnp.full((num_electrons, num_nuclei), 0.2)
    for neg_val in [-5.0, -20.0, -100.0]:
        cusp.decay_strengths[...] = jnp.full((num_nuclei,), neg_val)
        val = cusp(fixed_dists, charges)
        assert jnp.isfinite(val), f"Expected finite nuclei cusp value for raw param {neg_val}, got {val}"


def test_wavefunction_asymptotic_decay():
    """Moving any electron far from the nuclei strictly decreases log_psi toward -inf."""
    num_electrons = 3
    num_nuclei = 2
    rngs = nnx.Rngs(7)

    wf = WaveFunction(num_electrons=num_electrons, num_nuclei=num_nuclei, rngs=rngs)
    nuclei = Nucleus(
        position=jnp.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        charge=jnp.array([1.0, 2.0]),
    )
    base_positions = jnp.array([
        [0.1, 0.2, -0.1],
        [-0.3, -0.2, 0.4],
        [0.4, -0.1, 0.2],
    ])
    spins = jnp.array([[1.0], [-1.0], [1.0]])

    radii = [5.0, 20.0, 50.0]
    for electron_idx in range(num_electrons):
        log_psis = []
        for r in radii:
            pos = base_positions.at[electron_idx].set(jnp.array([r, 0.0, 0.0]))
            log_psi, _, _ = wf(Electron(position=pos, spin=spins), nuclei)
            assert jnp.isfinite(log_psi)
            log_psis.append(float(log_psi))

        assert log_psis[0] > log_psis[1] > log_psis[2], (
            f"Expected strictly decreasing log_psi for electron {electron_idx} at r={radii}, got {log_psis}"
        )
        # Verify decay is substantial (linear in r -> normalizable)
        assert log_psis[2] < -30.0, f"Expected large negative log_psi at r=50, got {log_psis[2]}"


def test_exponential_envelope_negative_params_finite():
    """ExponentialEnvelope remains finite even when raw pi/sigma parameters are negative."""
    envelope = ExponentialEnvelope(num_nuclei=2)
    dists = jnp.array([[0.5, 1.2], [2.0, 0.8]])
    for neg_val in [-5.0, -20.0, -100.0]:
        envelope.pi[...] = jnp.full((2,), neg_val)
        envelope.sigma[...] = jnp.full((2,), neg_val)
        val = envelope(dists)
        assert jnp.isfinite(val), f"Expected finite envelope for raw param {neg_val}, got {val}"


def test_wavefunction_kinetic_energy_and_grad_finite():
    """WaveFunction with updated cusps and envelope supports forward_laplacian and nnx.grad."""
    num_electrons = 2
    num_nuclei = 2
    rngs = nnx.Rngs(123)

    wf = WaveFunction(num_electrons=num_electrons, num_nuclei=num_nuclei, rngs=rngs)
    electrons = Electron(
        position=jnp.array([[0.3, 0.4, -0.2], [-0.4, 0.2, 0.5]]),
        spin=jnp.array([[1.0], [-1.0]]),
    )
    nuclei = Nucleus(
        position=jnp.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        charge=jnp.array([1.0, 1.0]),
    )

    ke = calculate_kinetic_energy(wf, electrons, nuclei)
    assert jnp.isfinite(ke), f"Expected finite kinetic energy, got {ke}"

    def loss_fn(model):
        log_psi, _, _ = model(electrons, nuclei)
        return log_psi

    grads = nnx.grad(loss_fn)(wf)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
