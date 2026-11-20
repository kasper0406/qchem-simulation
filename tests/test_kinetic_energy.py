from functools import partial

import chex
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qchem_simulation.electronic_optimization import (
    WaveFunction,
    calculate_kinetic_energy,
    calculate_kinetic_energy_full,
    calculate_kinetic_energy_per_electron,
    hamiltonian,
)
from qchem_simulation.utils import Electron, Nucleus


class GaussianWaveFunction(nnx.Module):
    """Analytical 3D isotropic Gaussian centered at nuclei.position[0]:
    log psi(r) = -alpha * sum_i ||r_i - R_0||^2
    """

    def __init__(self, alpha: float = 0.7):
        self.alpha = nnx.Param(jnp.array(alpha, dtype=jnp.float32))

    def __call__(self, electrons: Electron, nuclei: Nucleus):
        diff = electrons.position - nuclei.position[0]
        log_psi = -self.alpha * jnp.sum(diff ** 2)
        return log_psi, jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)

    def eval_wrt_electron_position(self, position, nuclei, position_index, all_positions, spins):
        modified_positions = all_positions.at[position_index].set(position)
        return self(Electron(position=modified_positions, spin=spins), nuclei)


class CorrelatedPairWaveFunction(nnx.Module):
    """Analytical multi-electron correlated wavefunction:
    log psi(r) = -alpha * sum_i ||r_i - R_0||^2 + beta * sum_{i < j} sqrt(||r_i - r_j||^2 + eps)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.35, eps: float = 0.25):
        self.alpha = nnx.Param(jnp.array(alpha, dtype=jnp.float32))
        self.beta = nnx.Param(jnp.array(beta, dtype=jnp.float32))
        self.eps = eps

    def __call__(self, electrons: Electron, nuclei: Nucleus):
        r = electrons.position
        one_body = -self.alpha * jnp.sum((r - nuclei.position[0]) ** 2)
        i, j = jnp.triu_indices(r.shape[0], k=1)
        pair_sq = jnp.sum((r[i] - r[j]) ** 2, axis=-1)
        two_body = self.beta * jnp.sum(jnp.sqrt(pair_sq + self.eps))
        return one_body + two_body, jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)

    def eval_wrt_electron_position(self, position, nuclei, position_index, all_positions, spins):
        modified_positions = all_positions.at[position_index].set(position)
        return self(Electron(position=modified_positions, spin=spins), nuclei)


def _make_system(num_electrons: int, num_nuclei: int, seed: int = 0):
    key = jax.random.key(seed)
    k_e, k_n = jax.random.split(key)
    e_pos = jax.random.normal(k_e, (num_electrons, 3), dtype=jnp.float32)
    e_spin = ((-1.0) ** jnp.arange(num_electrons, dtype=jnp.float32))[:, None]
    n_pos = jax.random.normal(k_n, (num_nuclei, 3), dtype=jnp.float32)
    n_charge = jnp.ones((num_nuclei,), dtype=jnp.float32) * (num_electrons / num_nuclei)
    return Electron(position=e_pos, spin=e_spin), Nucleus(position=n_pos, charge=n_charge)


def _neural_system_float64(num_electrons, num_nuclei, seed):
    wf = WaveFunction(num_electrons=num_electrons, num_nuclei=num_nuclei, rngs=nnx.Rngs(0))
    nnx.update(wf, jax.tree.map(lambda x: x.astype(jnp.float64), nnx.state(wf)))
    electrons, nuclei = _make_system(num_electrons, num_nuclei, seed)
    electrons, nuclei = jax.tree.map(lambda x: x.astype(jnp.float64), (electrons, nuclei))
    return wf, electrons, nuclei


# ---------------------------------------------------------------------------
# Analytical verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_electrons", [1, 3, 6])
def test_analytical_isotropic_gaussian(num_electrons: int):
    alpha = 0.7
    wf = GaussianWaveFunction(alpha=alpha)
    electrons, nuclei = _make_system(num_electrons, num_nuclei=1, seed=10 + num_electrons)

    diff = electrons.position - nuclei.position[0]
    expected = 3.0 * num_electrons * alpha - 2.0 * (alpha ** 2) * jnp.sum(diff ** 2)

    ke_per = calculate_kinetic_energy_per_electron(wf, electrons, nuclei)
    ke_full_0 = calculate_kinetic_energy_full(wf, electrons, nuclei, sparsity_threshold=0)
    ke_full_6 = calculate_kinetic_energy_full(wf, electrons, nuclei, sparsity_threshold=6)

    chex.assert_trees_all_close(ke_per, expected, rtol=1e-5, atol=1e-5)
    chex.assert_trees_all_close(ke_full_0, expected, rtol=1e-5, atol=1e-5)
    chex.assert_trees_all_close(ke_full_6, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_electrons", [2, 5])
def test_analytical_correlated_pair_wavefunction(num_electrons: int):
    alpha, beta, eps = 0.5, 0.35, 0.25
    wf = CorrelatedPairWaveFunction(alpha=alpha, beta=beta, eps=eps)
    electrons, nuclei = _make_system(num_electrons, num_nuclei=2, seed=20 + num_electrons)

    r = electrons.position
    # Exact gradient per electron
    diff_matrix = r[:, None, :] - r[None, :, :]  # (N, N, 3)
    sq_matrix = jnp.sum(diff_matrix ** 2, axis=-1)  # (N, N)
    s_matrix = jnp.sqrt(sq_matrix + eps)  # (N, N)
    # Mask diagonal (j == i contributes 0 since diff_matrix[i, i] == 0)
    grad_two_body = beta * jnp.sum(diff_matrix / s_matrix[:, :, None], axis=1)
    grad_log_psi = -2.0 * alpha * (r - nuclei.position[0]) + grad_two_body
    grad_sq = jnp.sum(grad_log_psi ** 2)

    # Exact Laplacian: each unordered pair (i < j) contributes twice
    i, j = jnp.triu_indices(num_electrons, k=1)
    pair_sq = jnp.sum((r[i] - r[j]) ** 2, axis=-1)
    lap_two_body = 2.0 * beta * jnp.sum((2.0 * pair_sq + 3.0 * eps) / ((pair_sq + eps) ** 1.5))
    lap_log_psi = -6.0 * num_electrons * alpha + lap_two_body

    expected = -0.5 * (lap_log_psi + grad_sq)

    ke_per = calculate_kinetic_energy_per_electron(wf, electrons, nuclei)
    ke_full_0 = calculate_kinetic_energy_full(wf, electrons, nuclei, sparsity_threshold=0)
    ke_full_6 = calculate_kinetic_energy_full(wf, electrons, nuclei, sparsity_threshold=6)

    chex.assert_trees_all_close(ke_per, expected, rtol=1e-5, atol=1e-5)
    chex.assert_trees_all_close(ke_full_0, expected, rtol=1e-5, atol=1e-5)
    chex.assert_trees_all_close(ke_full_6, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Numerical equivalence on neural WaveFunction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,num_electrons,num_nuclei",
    [
        ("H2", 2, 2),
        ("4e", 4, 2),
        ("H2O", 10, 3),
    ],
)
@jax.enable_x64()
def test_equivalence_on_wavefunction(name: str, num_electrons: int, num_nuclei: int):
    # Near-singular orbitals amplify float32 differences between contraction orders.
    wf, electrons, nuclei = _neural_system_float64(num_electrons, num_nuclei, seed=42)

    ke_per = calculate_kinetic_energy_per_electron(wf, electrons, nuclei)
    ke_full_0 = calculate_kinetic_energy_full(wf, electrons, nuclei, sparsity_threshold=0)
    ke_full_6 = calculate_kinetic_energy_full(wf, electrons, nuclei, sparsity_threshold=6)
    ke_default = calculate_kinetic_energy(wf, electrons, nuclei)
    ke_dispatch_per = calculate_kinetic_energy(wf, electrons, nuclei, method="per_electron")

    chex.assert_trees_all_close(ke_full_0, ke_per, rtol=1e-4, atol=1e-4)
    chex.assert_trees_all_close(ke_full_6, ke_per, rtol=1e-4, atol=1e-4)
    chex.assert_trees_all_close(ke_default, ke_full_0, rtol=1e-6, atol=1e-6)
    chex.assert_trees_all_close(ke_dispatch_per, ke_per, rtol=1e-6, atol=1e-6)


def test_invalid_method_raises():
    wf = GaussianWaveFunction()
    electrons, nuclei = _make_system(2, 1, seed=0)
    with pytest.raises(ValueError, match="Unknown kinetic energy method"):
        calculate_kinetic_energy(wf, electrons, nuclei, method="invalid")


# ---------------------------------------------------------------------------
# JAX transformations: nnx.jit, nnx.vmap, jax.grad wrt nuclei.position
# ---------------------------------------------------------------------------


@jax.enable_x64()
def test_jit_and_vmap_wavefunction():
    num_electrons, num_nuclei, batch_size = 4, 2, 5
    wf, electrons_single, nuclei = _neural_system_float64(num_electrons, num_nuclei, seed=7)

    # nnx.jit
    jitted_full = nnx.jit(calculate_kinetic_energy_full)
    ke_eager = calculate_kinetic_energy_full(wf, electrons_single, nuclei)
    ke_jit = jitted_full(wf, electrons_single, nuclei)
    chex.assert_trees_all_close(ke_jit, ke_eager, rtol=1e-5, atol=1e-5)

    # nnx.vmap over batch of electron samples
    batch_pos = jax.random.normal(
        jax.random.key(99), (batch_size, num_electrons, 3), dtype=jnp.float32
    ).astype(jnp.float64)
    batch_spin = jnp.tile(electrons_single.spin[None, :, :], (batch_size, 1, 1))
    electrons_batch = Electron(position=batch_pos, spin=batch_spin)

    vmapped_full = nnx.jit(nnx.vmap(calculate_kinetic_energy_full, in_axes=(None, 0, None, None)))
    vmapped_full_sparse = nnx.jit(
        nnx.vmap(partial(calculate_kinetic_energy_full, sparsity_threshold=6), in_axes=(None, 0, None))
    )
    vmapped_per = nnx.jit(nnx.vmap(calculate_kinetic_energy_per_electron, in_axes=(None, 0, None)))

    ke_batch_full = vmapped_full(wf, electrons_batch, nuclei)
    ke_batch_full_sparse = vmapped_full_sparse(wf, electrons_batch, nuclei)
    ke_batch_per = vmapped_per(wf, electrons_batch, nuclei)

    chex.assert_shape(ke_batch_full, (batch_size,))
    chex.assert_trees_all_close(ke_batch_full, ke_batch_per, rtol=1e-4, atol=1e-4)
    chex.assert_trees_all_close(ke_batch_full_sparse, ke_batch_per, rtol=1e-4, atol=1e-4)


def test_grad_wrt_nuclei_position_analytical():
    alpha = 0.7
    wf = GaussianWaveFunction(alpha=alpha)
    electrons, nuclei = _make_system(4, num_nuclei=2, seed=11)
    graphdef, state = nnx.split(wf)

    def kinetic_wrt_nuclei(nuclei_pos):
        merged_wf = nnx.merge(graphdef, state)
        return calculate_kinetic_energy_full(
            merged_wf, electrons, Nucleus(position=nuclei_pos, charge=nuclei.charge)
        )

    grad = jax.jit(jax.grad(kinetic_wrt_nuclei))(nuclei.position)
    # T_L = 3*N*alpha - 2*alpha^2 * sum_i ||r_i - R_0||^2
    # dT_L / dR_0 = 4*alpha^2 * sum_i (r_i - R_0), dT_L / dR_1 = 0
    expected_grad_0 = 4.0 * (alpha ** 2) * jnp.sum(electrons.position - nuclei.position[0], axis=0)
    expected_grad = jnp.stack([expected_grad_0, jnp.zeros((3,), dtype=jnp.float32)], axis=0)

    chex.assert_shape(grad, nuclei.position.shape)
    chex.assert_trees_all_close(grad, expected_grad, rtol=1e-5, atol=1e-5)


def test_grad_wrt_nuclei_position_wavefunction_and_hamiltonian():
    wf = WaveFunction(num_electrons=2, num_nuclei=2, rngs=nnx.Rngs(0))
    electrons, nuclei = _make_system(2, num_nuclei=2, seed=13)
    # Keep nuclei well separated from electrons for well-conditioned gradients
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=jnp.float32),
        charge=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    graphdef, state = nnx.split(wf)

    def energy_wrt_nuclei(nuclei_pos):
        merged_wf = nnx.merge(graphdef, state)
        e, _details = hamiltonian(
            merged_wf,
            electrons,
            Nucleus(position=nuclei_pos, charge=nuclei.charge),
            jax.random.key(0),
        )
        return e

    grad = jax.jit(jax.grad(energy_wrt_nuclei))(nuclei.position)
    chex.assert_shape(grad, nuclei.position.shape)
    assert bool(jnp.all(jnp.isfinite(grad)))

    # Sanity-check against central finite differences
    eps = 1e-3
    jitted_energy = jax.jit(energy_wrt_nuclei)
    fd = jnp.zeros_like(nuclei.position)
    for i in range(nuclei.position.shape[0]):
        for j in range(3):
            p_plus = nuclei.position.at[i, j].add(eps)
            p_minus = nuclei.position.at[i, j].add(-eps)
            fd = fd.at[i, j].set((jitted_energy(p_plus) - jitted_energy(p_minus)) / (2.0 * eps))
    chex.assert_trees_all_close(grad, fd, rtol=2e-2, atol=1e-2)
