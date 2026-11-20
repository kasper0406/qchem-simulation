import jax
import jax.numpy as jnp
import jax.scipy.special as jspecial
import optax
from flax import nnx

from qchem_simulation.electronic_optimization import WaveFunction
from qchem_simulation.nuclei_optimization import (
    nuclei_energy_grad_full,
    nuclei_energy_grad_hellmann_feynman,
    sample_and_optimize,
    simplified_hamiltonian,
)
from qchem_simulation.utils import Electron, Nucleus


class GaussianWaveFunction(nnx.Module):
    """
    Simple analytical 1-electron Gaussian wavefunction centered on nucleus 0:
        psi(x; R) = exp(-0.5 * alpha * ||x - R_0||^2)
    Because a Gaussian is not an exact eigenstate of the Coulomb Hamiltonian,
    its nuclear energy gradient has a large non-zero Pulay / score-function term.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = nnx.Param(jnp.array(alpha, dtype=jnp.float32))

    def __call__(self, electrons: Electron, nuclei: Nucleus):
        diff = electrons.position[0] - nuclei.position[0]
        log_psi = -0.5 * self.alpha * jnp.sum(jnp.square(diff))
        return log_psi, jnp.array(1.0, dtype=log_psi.dtype), jnp.array(0.0, dtype=log_psi.dtype)

    def eval_wrt_electron_position(self, position, nuclei, position_index, all_positions, spins):
        modified_positions = all_positions.at[position_index].set(position)
        electrons = Electron(position=modified_positions, spin=spins)
        return self(electrons, nuclei)


class FixedOriginGaussianWaveFunction(nnx.Module):
    """R-independent wavefunction (depends on x, not on nuclei.position) so grad_R log|psi| = 0."""

    def __call__(self, electrons: Electron, nuclei: Nucleus):
        log_psi = -0.5 * jnp.sum(jnp.square(electrons.position))
        return log_psi, jnp.array(1.0, dtype=log_psi.dtype), jnp.array(0.0, dtype=log_psi.dtype)

    def eval_wrt_electron_position(self, position, nuclei, position_index, all_positions, spins):
        modified_positions = all_positions.at[position_index].set(position)
        electrons = Electron(position=modified_positions, spin=spins)
        return self(electrons, nuclei)


def _exact_gaussian_2nuclei_energy(nuclei_positions: jax.Array, nuclei_charges: jax.Array, alpha: float) -> jax.Array:
    """
    Closed-form expectation value <E(R_0, R_1)> for psi(x; R) = exp(-0.5 * alpha * ||x - R_0||^2)
    with 1 electron and 2 nuclei at R_0, R_1 with charges Z_0, Z_1:
        <T>   = 3 * alpha / 2
        <V_0> = -2 * Z_0 * sqrt(alpha / pi)
        <V_1> = -Z_1 * erf(sqrt(alpha) * ||R_0 - R_1||) / ||R_0 - R_1||
        V_nn  = Z_0 * Z_1 / ||R_0 - R_1||
    """
    r01 = jnp.linalg.norm(nuclei_positions[0] - nuclei_positions[1])
    z0, z1 = nuclei_charges[0], nuclei_charges[1]
    kinetic = 1.5 * alpha
    v_e0 = -2.0 * z0 * jnp.sqrt(alpha / jnp.pi)
    v_e1 = -z1 * jspecial.erf(jnp.sqrt(alpha) * r01) / r01
    v_nn = (z0 * z1) / r01
    return kinetic + v_e0 + v_e1 + v_nn


def _finite_diff_grad(fn, x: jax.Array, eps: float = 1e-3) -> jax.Array:
    flat_x = x.reshape(-1)
    grads = []
    for i in range(flat_x.shape[0]):
        delta = jnp.zeros_like(flat_x).at[i].set(eps)
        f_plus = fn((flat_x + delta).reshape(x.shape))
        f_minus = fn((flat_x - delta).reshape(x.shape))
        grads.append((f_plus - f_minus) / (2.0 * eps))
    return jnp.stack(grads).reshape(x.shape)


def test_hellmann_feynman_clipping_not_distorted_by_charge_grad(capsys):
    """
    With a small nuclear charge Z = 1e-4 and electron distance r = 0.01:
      - ||dV/dR|| = Z / r^2 = 1.0  (well below max_norm = 10.0 -> must NOT be clipped)
      - ||dV/dZ|| = 1 / r   = 100  (way above max_norm = 10.0)
    Previously @clip_grad_global clipped across all arguments, shrinking dV/dR by ~10x.
    Also verifies no stray debug stdout is emitted.
    """
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([1e-4], dtype=jnp.float32),
    )
    electrons = Electron(
        position=jnp.array([[[0.0, 0.0, 0.01]]], dtype=jnp.float32),
        spin=jnp.array([[[1.0]]], dtype=jnp.float32),
    )

    grad = nuclei_energy_grad_hellmann_feynman(None, electrons, nuclei, jax.random.key(0))
    captured = capsys.readouterr()
    assert captured.out == ""

    # Exact d/dR_0 (-Z / ||x - R_0||) = -Z * (x - R_0) / ||x - R_0||^3 = (0, 0, -1.0)
    expected = jnp.array([[0.0, 0.0, -1.0]], dtype=jnp.float32)
    assert jnp.allclose(grad, expected, atol=1e-4, rtol=1e-4)


def test_hellmann_feynman_clips_large_nuclei_grad():
    """When ||dV/dR|| exceeds max_norm, it should be clipped to max_norm preserving direction."""
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([1.0], dtype=jnp.float32),
    )
    # r = 0.05 => ||dV/dR|| = 400.0 > max_norm = 10.0
    electrons = Electron(
        position=jnp.array([[[0.0, 0.0, 0.05]]], dtype=jnp.float32),
        spin=jnp.array([[[1.0]]], dtype=jnp.float32),
    )

    grad = nuclei_energy_grad_hellmann_feynman(None, electrons, nuclei, jax.random.key(0), max_norm=10.0)
    expected = jnp.array([[0.0, 0.0, -10.0]], dtype=jnp.float32)
    assert jnp.allclose(grad, expected, atol=1e-4, rtol=1e-4)


def test_full_grad_clipping_not_distorted_by_electron_grad():
    """
    Place two electrons very close to each other (r_ee = 1e-3 => ||dE_L/dx|| ~ 1e6)
    while far from the nucleus (r_en = 1.0 => ||dE_L/dR|| = 2.0 < 10.0).
    With a constant (R-independent) wavefunction, the Pulay term is zero and
    nuclei_energy_grad_full must return the exact unclipped dE_L/dR without distortion
    from the huge electron-electron gradient.
    """
    wf = FixedOriginGaussianWaveFunction()
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([1.0], dtype=jnp.float32),
    )
    electrons = Electron(
        position=jnp.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.001]]], dtype=jnp.float32),
        spin=jnp.array([[[1.0], [-1.0]]], dtype=jnp.float32),
    )

    grad = nuclei_energy_grad_full(wf, electrons, nuclei, jax.random.key(0), max_norm=10.0)

    # Exact d/dR_0 (-1/||x_0 - R_0|| - 1/||x_1 - R_0||) = (0, 0, -(1/1.0^2 + 1/1.001^2))
    expected_z = -(1.0 / (1.0**2) + 1.0 / (1.001**2))
    expected = jnp.array([[0.0, 0.0, expected_z]], dtype=jnp.float32)
    assert jnp.allclose(grad, expected, atol=1e-4, rtol=1e-4)


def test_full_grad_matches_finite_difference_with_pulay_term(monkeypatch):
    """
    Verify nuclei_energy_grad_full matches finite-difference d<E(R)>/dR on an analytical
    R-dependent Gaussian wavefunction centered at nucleus 0 in a 2-nucleus system.

    Because the Gaussian is not an eigenstate of the Coulomb Hamiltonian, the Pulay term
    2 * E[(E_L - <E_L>) * grad_R log|psi|] contributes the entire d<V_{e,1}>/dR_0 force.
    Omitting the Pulay term gives ~+0.444 on R_0_z instead of the true ~+0.073.
    """
    alpha = 1.0
    wf = GaussianWaveFunction(alpha=alpha)
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=jnp.float32),
        charge=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )

    # Exact i.i.d. samples from |psi(x; R)|^2 = N(R_0, 1 / (2 * alpha) * I_3)
    # Using antithetic variates (R_0 + eps, R_0 - eps) sharply reduces MC variance.
    half_n = 4000
    sigma = 1.0 / jnp.sqrt(2.0 * alpha)
    eps = sigma * jax.random.normal(jax.random.key(42), shape=(half_n, 1, 3), dtype=jnp.float32)
    pos = jnp.concatenate([nuclei.position[0] + eps, nuclei.position[0] - eps], axis=0)
    spins = jnp.ones((2 * half_n, 1, 1), dtype=jnp.float32)
    electron_samples = Electron(position=pos, spin=spins)

    # Feed exact samples into the geometry driver to isolate the gradient from MCMC error.
    def exact_sampler(**kwargs):
        samples = jax.tree.map(lambda x: x[:, None, ...], electron_samples)
        return samples, None, None

    monkeypatch.setattr("qchem_simulation.nuclei_optimization.sample_from_wavefunction", exact_sampler)
    optimizer = optax.sgd(1.0)
    updated_nuclei, _ = sample_and_optimize(
        wf, optimizer, optimizer.init(nuclei.position), nuclei,
        num_electron_samples=2 * half_n, num_electron_chains=1,
        sum_of_charges=1, key=jax.random.key(7),
    )
    mc_grad = nuclei.position - updated_nuclei.position

    fd_grad = _finite_diff_grad(
        lambda r: _exact_gaussian_2nuclei_energy(r, nuclei.charge, alpha),
        nuclei.position,
        eps=1e-3,
    )

    # Confirm the Pulay contribution is large (naive E[grad_R E_L] on R_0_z is +1/1.5^2 = +0.4444).
    assert abs(float(fd_grad[0, 2]) - (1.0 / 1.5**2)) > 0.3

    # The default geometry update should match finite-difference d<E>/dR closely.
    assert jnp.allclose(mc_grad, fd_grad, atol=2e-2, rtol=5e-2), (
        f"MC grad {mc_grad} did not match finite-difference grad {fd_grad}"
    )


def test_full_grad_batching_invariance():
    """
    Verify batching (including non-divisible batch_size) produces the exact same result
    as computing over all samples in a single batch, confirming <E_L> centering across
    batches via linearity of expectation.
    """
    wf = GaussianWaveFunction(alpha=1.0)
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=jnp.float32),
        charge=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )

    num_samples = 73
    pos = jax.random.normal(jax.random.key(0), shape=(num_samples, 1, 3), dtype=jnp.float32)
    spins = jnp.ones((num_samples, 1, 1), dtype=jnp.float32)
    electron_samples = Electron(position=pos, spin=spins)

    grad_single_batch = nuclei_energy_grad_full(wf, electron_samples, nuclei, jax.random.key(1), batch_size=num_samples)
    grad_multi_batch = nuclei_energy_grad_full(wf, electron_samples, nuclei, jax.random.key(1), batch_size=20)
    grad_small_batch = nuclei_energy_grad_full(wf, electron_samples, nuclei, jax.random.key(1), batch_size=7)

    assert jnp.allclose(grad_single_batch, grad_multi_batch, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(grad_single_batch, grad_small_batch, atol=1e-5, rtol=1e-5)


def test_full_grad_works_with_neural_wavefunction_under_jit():
    """Smoke test that nuclei_energy_grad_full works under nnx.jit with the full WaveFunction."""
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=jnp.float32),
        charge=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    wf = WaveFunction(num_electrons=2, num_nuclei=2, rngs=nnx.Rngs(0))

    num_samples = 4
    pos = jax.random.normal(jax.random.key(3), shape=(num_samples, 2, 3), dtype=jnp.float32) + 0.5
    spins = jnp.tile(jnp.array([[1.0], [-1.0]], dtype=jnp.float32)[None, :, :], (num_samples, 1, 1))
    electron_samples = Electron(position=pos, spin=spins)

    jit_grad_fn = nnx.jit(nuclei_energy_grad_full)
    grad = jit_grad_fn(wf, electron_samples, nuclei, jax.random.key(4))

    assert grad.shape == (2, 3)
    assert jnp.all(jnp.isfinite(grad))


@jax.enable_x64()
def test_coulomb_geometry_derivative_near_and_at_contact():
    electrons = Electron(position=jnp.zeros((1, 3)), spin=jnp.ones((1, 1)))

    def energy(x):
        nuclei = Nucleus(position=jnp.array([[x, 0.0, 0.0]]), charge=jnp.ones(1))
        return simplified_hamiltonian(electrons, nuclei)

    radius = 1e-8
    value, gradient = jax.value_and_grad(energy)(radius)
    assert jnp.isclose(value, -1 / radius, rtol=1e-12)
    assert jnp.isclose(gradient, 1 / radius**2, rtol=1e-12)
    contact_value, contact_gradient = jax.value_and_grad(energy)(0.0)
    assert contact_value == -1e12
    assert contact_gradient == 0
