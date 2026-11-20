import time
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from qchem_simulation.electronic_optimization import (
    WaveFunction,
    WalkerState,
    adapt_step_size,
    init_walkers,
    step_walkers,
    train_wavefunction,
)
from qchem_simulation.utils import Electron, Nucleus


class GaussianWaveFunction(nnx.Module):
    """Analytical Gaussian wavefunction for testing MCMC sampling.

    Returns log_psi(x) = -0.25 * sum((x - mean)^2 / std^2) so that
    |psi(x)|^2 = exp(2 * log_psi(x)) is an exact Gaussian N(mean, std^2 I)
    for every electron coordinate.
    """

    def __init__(self, mean: jax.Array, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, electrons: Electron, nuclei: Nucleus):
        diff = electrons.position - self.mean
        log_psi = -0.25 * jnp.sum((diff / self.std) ** 2)
        sign = jnp.array(1.0, dtype=jnp.float32)
        ortho_measure = jnp.array(0.0, dtype=jnp.float32)
        return log_psi, sign, ortho_measure


def test_walker_state_pytree_and_init_walkers():
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([2], dtype=jnp.int32),
    )
    walkers = init_walkers(
        nuclei=nuclei,
        num_chains=8,
        rng=jax.random.key(0),
        sum_of_charges=2,
        init_step_size=0.25,
    )

    assert isinstance(walkers, WalkerState)
    assert walkers.position.shape == (8, 2, 3)
    assert walkers.spin.shape == (8, 2, 1)
    assert walkers.step_size.shape == ()
    assert jnp.isclose(walkers.step_size, 0.25)

    # Verify JAX pytree round-trip
    leaves, treedef = jax.tree_util.tree_flatten(walkers)
    assert len(leaves) == 3
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.allclose(reconstructed.position, walkers.position)
    assert jnp.allclose(reconstructed.spin, walkers.spin)
    assert jnp.allclose(reconstructed.step_size, walkers.step_size)


def test_adapt_step_size_bounds_and_direction():
    # High acceptance -> step size increases
    increased = adapt_step_size(0.1, acceptance_rate=0.9, target_acceptance=0.5, adapt_rate=0.1)
    assert float(increased) > 0.1

    # Low acceptance -> step size decreases
    decreased = adapt_step_size(0.1, acceptance_rate=0.1, target_acceptance=0.5, adapt_rate=0.1)
    assert float(decreased) < 0.1

    # Clamping to [min_step_size, max_step_size]
    clamped_low = adapt_step_size(1e-4, acceptance_rate=0.0, target_acceptance=0.5, adapt_rate=1.0, min_step_size=1e-4)
    assert jnp.isclose(clamped_low, 1e-4)

    clamped_high = adapt_step_size(2.0, acceptance_rate=1.0, target_acceptance=0.5, adapt_rate=1.0, max_step_size=2.0)
    assert jnp.isclose(clamped_high, 2.0)

    # adapt_rate == 0.0 preserves step_size without clamping
    unchanged = adapt_step_size(5.0, acceptance_rate=0.0, target_acceptance=0.5, adapt_rate=0.0, max_step_size=2.0)
    assert jnp.isclose(unchanged, 5.0)


def test_adaptive_step_size_increases_from_tiny_initial():
    """Tiny initial step_size=1e-4 gives ~1.0 acceptance and should adapt upward toward target_acceptance=0.5."""
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([2], dtype=jnp.int32),
    )
    # Tighter Gaussian (std=0.5) so optimal MALA step size lies well inside [1e-4, 2.0]
    wf = GaussianWaveFunction(mean=jnp.zeros((2, 3), dtype=jnp.float32), std=0.5)
    walkers = init_walkers(nuclei, num_chains=16, rng=jax.random.key(1), sum_of_charges=2, init_step_size=1e-4)

    key = jax.random.key(2)
    # Run adaptation sweeps
    _, walkers, _, _ = step_walkers(
        wf, nuclei, walkers, key, num_steps=300, target_acceptance=0.5, adapt_rate=0.1
    )
    # Measure steady-state acceptance rate after adaptation
    _, walkers_after, _, final_acc = step_walkers(
        wf, nuclei, walkers, jax.random.key(3), num_steps=50, target_acceptance=0.5, adapt_rate=0.05
    )

    assert float(walkers.step_size) > 1e-2, f"Expected step_size to grow from 1e-4, got {float(walkers.step_size)}"
    assert 0.35 <= float(final_acc) <= 0.65, f"Expected acceptance near 0.5, got {float(final_acc)}"
    assert jnp.isfinite(walkers_after.step_size)


def test_adaptive_step_size_decreases_from_huge_initial():
    """Huge initial step_size=5.0 gives very low acceptance and should adapt downward toward target_acceptance=0.5."""
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([2], dtype=jnp.int32),
    )
    wf = GaussianWaveFunction(mean=jnp.zeros((2, 3), dtype=jnp.float32), std=0.5)
    walkers = init_walkers(nuclei, num_chains=16, rng=jax.random.key(10), sum_of_charges=2, init_step_size=5.0)

    _, walkers, _, _ = step_walkers(
        wf, nuclei, walkers, jax.random.key(11), num_steps=150, target_acceptance=0.5, adapt_rate=0.1
    )
    _, _, _, final_acc = step_walkers(
        wf, nuclei, walkers, jax.random.key(12), num_steps=50, target_acceptance=0.5, adapt_rate=0.05
    )

    assert float(walkers.step_size) < 1.5, f"Expected step_size to drop from 5.0, got {float(walkers.step_size)}"
    assert 0.35 <= float(final_acc) <= 0.65, f"Expected acceptance near 0.5, got {float(final_acc)}"


def test_persistent_walkers_sample_analytical_gaussian():
    """Persistent walkers across multiple step_walkers calls accurately sample N(mu, sigma^2 I) without re-burnin."""
    target_mean = jnp.array([[1.5, -0.5, 0.75], [-1.0, 0.8, -0.25]], dtype=jnp.float32)
    target_std = 0.7
    wf = GaussianWaveFunction(mean=target_mean, std=target_std)

    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([2], dtype=jnp.int32),
    )
    walkers = init_walkers(nuclei, num_chains=32, rng=jax.random.key(42), sum_of_charges=2, init_step_size=0.2)

    # Single initial burn-in with adaptation
    _, walkers, _, _ = step_walkers(
        wf, nuclei, walkers, jax.random.key(43), num_steps=100, target_acceptance=0.5, adapt_rate=0.05
    )

    # Collect samples across 10 consecutive persistent calls (no re-init, no re-burnin)
    collected = []
    keys = jax.random.split(jax.random.key(44), 10)
    for k in keys:
        samples, walkers, _, _ = step_walkers(
            wf, nuclei, walkers, k, num_steps=10, target_acceptance=0.5, adapt_rate=0.0
        )
        # Verify final sample matches carried walker position
        assert jnp.allclose(samples.position[-1], walkers.position)
        collected.append(samples.position)

    all_positions = jnp.concatenate(collected, axis=0)  # (100, 32, 2, 3)
    emp_mean = jnp.mean(all_positions, axis=(0, 1))
    emp_var = jnp.var(all_positions, axis=(0, 1))

    assert jnp.allclose(emp_mean, target_mean, atol=0.12), f"Empirical mean {emp_mean} != {target_mean}"
    assert jnp.allclose(emp_var, target_std ** 2, atol=0.12), f"Empirical var {emp_var} != {target_std ** 2}"


def test_train_wavefunction_persistent_walkers_and_speedup():
    """Verify train_wavefunction runs cleanly with persistent walkers and is faster than cold-start."""
    # H2 molecule (2 nuclei, 2 electrons)
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=jnp.float32),
        charge=jnp.array([1, 1], dtype=jnp.int32),
    )

    def make_model_and_opt():
        wf = WaveFunction(
            num_electrons=2,
            num_nuclei=2,
            rngs=nnx.Rngs(0),
        )
        opt = nnx.ModelAndOptimizer(
            wf,
            optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3)),
        )
        return wf, opt

    num_chains = 8
    kept_steps_per_opt = 5  # 5 * 8 = 40 samples per optimization step
    batch_size = 40
    burnin_steps = 40

    # --- Persistent walkers ---
    wf_p, opt_p = make_model_and_opt()
    # Warmup JIT (1 step)
    train_wavefunction(
        wf_p,
        opt_p,
        nuclei,
        num_steps=1,
        num_samples=kept_steps_per_opt,
        num_chains=num_chains,
        key=jax.random.key(100),
        persistent_walkers=True,
        initial_burnin_steps=burnin_steps,
        mcmc_thinning_factor=1,
        batch_size=batch_size,
    )
    t0 = time.perf_counter()
    final_walkers = train_wavefunction(
        wf_p,
        opt_p,
        nuclei,
        num_steps=6,
        num_samples=kept_steps_per_opt,
        num_chains=num_chains,
        key=jax.random.key(101),
        persistent_walkers=True,
        initial_burnin_steps=burnin_steps,
        mcmc_thinning_factor=1,
        batch_size=batch_size,
    )
    jax.block_until_ready(final_walkers.position)
    persistent_time = time.perf_counter() - t0

    assert isinstance(final_walkers, WalkerState)
    assert final_walkers.position.shape == (num_chains, 2, 3)
    assert jnp.all(jnp.isfinite(final_walkers.position))
    assert jnp.isfinite(final_walkers.step_size)

    # --- Cold-start walkers (same 40 kept samples per step, re-burning 40 steps every opt step) ---
    wf_c, opt_c = make_model_and_opt()
    cold_num_samples = burnin_steps + kept_steps_per_opt
    # Warmup JIT (1 step)
    train_wavefunction(
        wf_c,
        opt_c,
        nuclei,
        num_steps=1,
        num_samples=cold_num_samples,
        num_chains=num_chains,
        key=jax.random.key(200),
        persistent_walkers=False,
        initial_burnin_steps=burnin_steps,
        mcmc_thinning_factor=1,
        batch_size=batch_size,
    )
    t0 = time.perf_counter()
    cold_res = train_wavefunction(
        wf_c,
        opt_c,
        nuclei,
        num_steps=6,
        num_samples=cold_num_samples,
        num_chains=num_chains,
        key=jax.random.key(201),
        persistent_walkers=False,
        initial_burnin_steps=burnin_steps,
        mcmc_thinning_factor=1,
        batch_size=batch_size,
    )
    cold_time = time.perf_counter() - t0

    assert cold_res is None
    speedup = cold_time / persistent_time
    print(f"\nBenchmark (6 opt steps, 40 samples/step, burnin=40): persistent={persistent_time:.3f}s, cold-start={cold_time:.3f}s, speedup={speedup:.2f}x")
    assert persistent_time < cold_time, (
        f"Expected persistent walkers ({persistent_time:.3f}s) to be faster than cold-start ({cold_time:.3f}s)"
    )
