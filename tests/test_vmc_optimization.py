import chex
import jax
import jax.numpy as jnp
import pytest
import optax
from flax import nnx

from qchem_simulation.electronic_optimization import (
    WaveFunction,
    local_energy,
    optimize_wave_function_batched,
    optimize_wave_function_full,
)
from qchem_simulation.utils import Electron, Nucleus


class HydrogenTrialWaveFunction(nnx.Module):
    """Exact-form 1-electron hydrogen-like trial wavefunction psi(r) = exp(-alpha * |r - R|)."""

    def __init__(self, alpha: float):
        self.alpha = nnx.Param(jnp.array(alpha, dtype=jnp.float32))

    def __call__(self, electrons: Electron, nuclei: Nucleus):
        r = jnp.linalg.norm(electrons.position[0] - nuclei.position[0])
        log_psi = -self.alpha * r
        return log_psi.squeeze(), jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)

    def eval_wrt_electron_position(self, position, nuclei: Nucleus, position_index: int, all_positions, spins):
        modified_positions = all_positions.at[position_index].set(position)
        electrons = Electron(position=modified_positions, spin=spins)
        return self(electrons, nuclei)


def _sample_hydrogen_exact(alpha: float, num_samples: int, key: jax.Array) -> Electron:
    """Draw exact i.i.d. samples from |psi(r)|^2 proportional to exp(-2 * alpha * |r|).

    In 3D spherical coordinates, |r| ~ Gamma(shape=3, rate=2*alpha) and the
    direction is uniform on the unit sphere S^2.
    """
    r_key, dir_key = jax.random.split(key)
    radii = jax.random.gamma(r_key, a=3.0, shape=(num_samples, 1, 1)) / (2.0 * alpha)
    directions = jax.random.normal(dir_key, shape=(num_samples, 1, 3))
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    positions = radii * directions
    spins = jnp.ones((num_samples, 1, 1), dtype=jnp.float32)
    return Electron(position=positions, spin=spins)


def _hydrogen_nucleus() -> Nucleus:
    return Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([1.0], dtype=jnp.float32),
    )


@pytest.mark.parametrize("batch_size", [10, 13, 100])
def test_full_and_batched_match_default_hydrogen(batch_size):
    """optimize_wave_function_full and optimize_wave_function_batched must produce
    identical average energy, hamiltonian details, log_psi, and parameter updates."""
    nuclei = _hydrogen_nucleus()
    samples = _sample_hydrogen_exact(alpha=1.4, num_samples=40, key=jax.random.key(0))

    wf_full = HydrogenTrialWaveFunction(alpha=1.4)
    wf_batched = HydrogenTrialWaveFunction(alpha=1.4)
    opt_full = nnx.ModelAndOptimizer(wf_full, optax.sgd(0.1))
    opt_batched = nnx.ModelAndOptimizer(wf_batched, optax.sgd(0.1))

    step_key = jax.random.key(42)
    e_full, h_full, lp_full = optimize_wave_function_full(
        wf_full, opt_full, samples, nuclei, step_key
    )
    e_batched, h_batched, lp_batched = optimize_wave_function_batched(
        wf_batched, opt_batched, samples, nuclei, step_key, batch_size=batch_size
    )

    chex.assert_trees_all_close(e_full, e_batched, rtol=1e-5, atol=1e-5)
    chex.assert_trees_all_close(h_full, h_batched, rtol=1e-5, atol=1e-5)
    chex.assert_trees_all_close(lp_full, lp_batched, rtol=1e-5, atol=1e-5)
    chex.assert_trees_all_close(
        nnx.state(wf_full, nnx.Param),
        nnx.state(wf_batched, nnx.Param),
        rtol=1e-5,
        atol=1e-5,
    )


def test_full_and_batched_match_with_nonzero_lambda_virial():
    """Verify gradient centering consistency between full and batched even when
    lambda_virial != 0."""
    nuclei = _hydrogen_nucleus()
    samples = _sample_hydrogen_exact(alpha=1.4, num_samples=40, key=jax.random.key(1))

    wf_full = HydrogenTrialWaveFunction(alpha=1.4)
    wf_batched = HydrogenTrialWaveFunction(alpha=1.4)
    opt_full = nnx.ModelAndOptimizer(wf_full, optax.sgd(0.05))
    opt_batched = nnx.ModelAndOptimizer(wf_batched, optax.sgd(0.05))

    step_key = jax.random.key(7)
    e_full, h_full, lp_full = optimize_wave_function_full(
        wf_full, opt_full, samples, nuclei, step_key, lambda_virial=1e-2
    )
    e_batched, h_batched, lp_batched = optimize_wave_function_batched(
        wf_batched, opt_batched, samples, nuclei, step_key, batch_size=10, lambda_virial=1e-2
    )

    chex.assert_trees_all_close(e_full, e_batched, rtol=1e-5, atol=1e-5)
    chex.assert_trees_all_close(h_full, h_batched, rtol=1e-5, atol=1e-5)
    chex.assert_trees_all_close(lp_full, lp_batched, rtol=1e-5, atol=1e-5)
    chex.assert_trees_all_close(
        nnx.state(wf_full, nnx.Param),
        nnx.state(wf_batched, nnx.Param),
        rtol=1e-5,
        atol=1e-5,
    )


@jax.enable_x64()
def test_full_and_batched_match_neural_wavefunction():
    """Compare neural VMC gradients through unit SGD updates."""
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float64),
        charge=jnp.array([2.0], dtype=jnp.float64),
    )
    num_samples = 12
    pos = jax.random.normal(
        jax.random.key(10), shape=(num_samples, 2, 3), dtype=jnp.float32
    ).astype(jnp.float64) * 0.5
    spins = jnp.tile(jnp.array([[1.0], [-1.0]], dtype=jnp.float64)[None, :, :], (num_samples, 1, 1))
    samples = Electron(position=pos, spin=spins)

    wf_full = WaveFunction(num_electrons=2, num_nuclei=1, rngs=nnx.Rngs(123))
    nnx.update(wf_full, jax.tree.map(lambda p: p.astype(jnp.float64), nnx.state(wf_full, nnx.Param)))
    wf_batched = nnx.clone(wf_full)

    initial_params = jax.tree.map(jnp.copy, nnx.state(wf_full, nnx.Param))
    opt_full = nnx.ModelAndOptimizer(wf_full, optax.sgd(1.0))
    opt_batched = nnx.ModelAndOptimizer(wf_batched, optax.sgd(1.0))

    step_key = jax.random.key(99)
    e_full, h_full, lp_full = optimize_wave_function_full(
        wf_full, opt_full, samples, nuclei, step_key
    )
    e_batched, h_batched, lp_batched = optimize_wave_function_batched(
        wf_batched, opt_batched, samples, nuclei, step_key, batch_size=4
    )

    chex.assert_trees_all_close(e_full, e_batched, rtol=1e-7, atol=1e-8)
    chex.assert_trees_all_close(h_full, h_batched, rtol=1e-7, atol=1e-8)
    chex.assert_trees_all_close(lp_full, lp_batched, rtol=1e-7, atol=1e-8)
    update_full = jax.tree.map(jnp.subtract, nnx.state(wf_full, nnx.Param), initial_params)
    update_batched = jax.tree.map(jnp.subtract, nnx.state(wf_batched, nnx.Param), initial_params)
    chex.assert_trees_all_close(update_full, update_batched, rtol=1e-7, atol=1e-8)


def test_vmc_gradient_matches_exact_score_function_formula():
    """Default optimization step must compute E[(E_L - <E_L>) * grad_theta log|psi|^2]
    without any virial penalty bias."""
    nuclei = _hydrogen_nucleus()
    samples = _sample_hydrogen_exact(alpha=1.5, num_samples=64, key=jax.random.key(2))

    wf = HydrogenTrialWaveFunction(alpha=1.5)
    lr = 0.1
    opt = nnx.ModelAndOptimizer(wf, optax.sgd(lr))

    # Compute the reference unbiased VMC gradient directly from local_energy
    keys = jax.random.split(jax.random.key(3), 64)
    local_energies, log_grads, _, _ = nnx.vmap(local_energy, in_axes=(None, 0, None, 0))(
        wf, samples, nuclei, keys
    )
    centered = local_energies - jnp.mean(local_energies)
    expected_grad = jnp.mean(centered * log_grads["alpha"].value)
    expected_alpha = 1.5 - lr * expected_grad

    optimize_wave_function_batched(
        wf, opt, samples, nuclei, jax.random.key(3), batch_size=16
    )

    chex.assert_trees_all_close(wf.alpha.value, expected_alpha, rtol=1e-5, atol=1e-5)


def test_exact_ground_state_has_zero_gradient_no_virial_bias():
    """At the exact hydrogen ground state (alpha = 1.0, E = -0.5 Ha), the local
    energy is identically -0.5 everywhere, so an unbiased VMC step must produce
    zero parameter update. A pointwise virial penalty (2*T_L + V)^2 would spuriously
    push alpha away from 1.0."""
    nuclei = _hydrogen_nucleus()
    samples = _sample_hydrogen_exact(alpha=1.0, num_samples=100, key=jax.random.key(4))

    wf = HydrogenTrialWaveFunction(alpha=1.0)
    opt = nnx.ModelAndOptimizer(wf, optax.sgd(0.1))

    avg_energy, _, _ = optimize_wave_function_batched(
        wf, opt, samples, nuclei, jax.random.key(5), batch_size=25
    )

    chex.assert_trees_all_close(avg_energy, jnp.array(-0.5, dtype=jnp.float32), atol=1e-5)
    chex.assert_trees_all_close(wf.alpha.value, jnp.array(1.0, dtype=jnp.float32), atol=1e-5)


def test_vmc_gradient_sign_and_convergence_to_hydrogen_ground_state():
    """For psi(r) = exp(-alpha * |r|), E(alpha) = alpha^2 / 2 - alpha has
    dE/dalpha = alpha - 1 and minimum at alpha = 1.0, E = -0.5 Ha.
    Verify both gradient signs (alpha > 1 decreases, alpha < 1 increases) and
    convergence toward the exact ground state."""
    nuclei = _hydrogen_nucleus()

    # Check gradient sign for alpha > 1 (dE/dalpha > 0 => alpha should decrease)
    wf_high = HydrogenTrialWaveFunction(alpha=1.5)
    opt_high = nnx.ModelAndOptimizer(wf_high, optax.sgd(0.1))
    samples_high = _sample_hydrogen_exact(alpha=1.5, num_samples=500, key=jax.random.key(11))
    optimize_wave_function_batched(wf_high, opt_high, samples_high, nuclei, jax.random.key(12), batch_size=100)
    assert float(wf_high.alpha.value) < 1.5, f"Expected alpha to decrease from 1.5, got {wf_high.alpha.value}"

    # Check gradient sign for alpha < 1 (dE/dalpha < 0 => alpha should increase)
    wf_low = HydrogenTrialWaveFunction(alpha=0.6)
    opt_low = nnx.ModelAndOptimizer(wf_low, optax.sgd(0.1))
    samples_low = _sample_hydrogen_exact(alpha=0.6, num_samples=500, key=jax.random.key(13))
    optimize_wave_function_batched(wf_low, opt_low, samples_low, nuclei, jax.random.key(14), batch_size=100)
    assert float(wf_low.alpha.value) > 0.6, f"Expected alpha to increase from 0.6, got {wf_low.alpha.value}"

    # Multi-step VMC optimization from alpha = 1.5 toward alpha = 1.0, E = -0.5 Ha
    wf = HydrogenTrialWaveFunction(alpha=1.5)
    opt = nnx.ModelAndOptimizer(wf, optax.sgd(0.25))
    key = jax.random.key(20)
    energies = []
    for _ in range(25):
        key, sample_key, step_key = jax.random.split(key, 3)
        samples = _sample_hydrogen_exact(alpha=float(wf.alpha.value), num_samples=400, key=sample_key)
        avg_e, _, _ = optimize_wave_function_batched(wf, opt, samples, nuclei, step_key, batch_size=100)
        energies.append(float(avg_e))

    assert energies[-1] < energies[0], f"Energy did not decrease: {energies[0]} -> {energies[-1]}"
    assert abs(float(wf.alpha.value) - 1.0) < 0.05, f"alpha did not converge near 1.0: {wf.alpha.value}"
    assert abs(energies[-1] - (-0.5)) < 0.02, f"Energy did not converge near -0.5 Ha: {energies[-1]}"
