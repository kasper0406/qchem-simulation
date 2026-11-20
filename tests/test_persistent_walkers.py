import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from qchem_simulation.electronic_optimization import (
    WalkerState,
    adapt_step_size,
    init_walkers,
    optimize_wave_function_batched,
    sample_from_wavefunction,
    step_walkers,
    train_wavefunction,
)
from qchem_simulation.utils import Electron, Nucleus
from .test_vmc_optimization import HydrogenTrialWaveFunction


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
        wf, nuclei, walkers, key, num_steps=300, mcmc_burnin_steps=300, target_acceptance=0.5, adapt_rate=0.1
    )
    # Measure steady-state acceptance rate after adaptation
    _, walkers_after, _, final_acc = step_walkers(
        wf, nuclei, walkers, jax.random.key(3), num_steps=50, target_acceptance=0.5, adapt_rate=0.05
    )

    assert float(walkers.step_size) > 1e-2, f"Expected step_size to grow from 1e-4, got {float(walkers.step_size)}"
    assert 0.35 <= float(final_acc) <= 0.65, f"Expected acceptance near 0.5, got {float(final_acc)}"
    assert walkers_after.step_size == walkers.step_size


def test_adaptive_step_size_decreases_from_huge_initial():
    """Huge initial step_size=5.0 gives very low acceptance and should adapt downward toward target_acceptance=0.5."""
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([2], dtype=jnp.int32),
    )
    wf = GaussianWaveFunction(mean=jnp.zeros((2, 3), dtype=jnp.float32), std=0.5)
    walkers = init_walkers(nuclei, num_chains=16, rng=jax.random.key(10), sum_of_charges=2, init_step_size=5.0)

    _, walkers, _, _ = step_walkers(
        wf, nuclei, walkers, jax.random.key(11), num_steps=150, mcmc_burnin_steps=150, target_acceptance=0.5, adapt_rate=0.1
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
        wf, nuclei, walkers, jax.random.key(43), num_steps=100, mcmc_burnin_steps=100, target_acceptance=0.5, adapt_rate=0.05
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


@pytest.mark.parametrize("persistent", [False, True])
def test_train_wavefunction_with_partial_batches(persistent):
    nuclei = Nucleus(position=jnp.zeros((1, 3)), charge=jnp.ones(1))
    wf = HydrogenTrialWaveFunction(alpha=1.2)
    wf.num_up = 0
    optimizer = nnx.ModelAndOptimizer(wf, optax.sgd(0.01))
    key = jax.random.key(200)
    options = dict(init_step_size=0.03, target_acceptance=0.2, adapt_rate=0.3)
    result = train_wavefunction(
        wf, optimizer, nuclei, num_steps=2, num_samples=5, num_chains=3, key=key,
        persistent_walkers=persistent, initial_burnin_steps=2, mcmc_thinning_factor=1,
        batch_size=4, **options,
    )
    assert jnp.isfinite(wf.alpha[...])
    assert not jnp.isclose(wf.alpha[...], 1.2)
    if persistent:
        init_key, warmup_key, _ = jax.random.split(key, 3)
        initial = init_walkers(nuclei, 3, init_key, 1, options["init_step_size"], num_up=0)
        _, warmed, _, _ = step_walkers(
            HydrogenTrialWaveFunction(alpha=1.2), nuclei, initial, warmup_key,
            num_steps=2, mcmc_burnin_steps=2, target_acceptance=options["target_acceptance"],
            adapt_rate=options["adapt_rate"],
        )
        assert jnp.all(result.spin == -1)
        assert result.position.shape == (3, 1, 3)
        assert jnp.isfinite(result.position).all()
        assert result.step_size == warmed.step_size
        assert result.step_size != options["init_step_size"]
    else:
        assert result is None
        reference = HydrogenTrialWaveFunction(alpha=1.2)
        reference.num_up = 0
        reference_opt = nnx.ModelAndOptimizer(reference, optax.sgd(0.01))
        _, loop_key = jax.random.split(key)
        for iteration_key in jax.random.split(loop_key, 2):
            sample_key, energy_key = jax.random.split(iteration_key)
            samples, _, _ = sample_from_wavefunction(
                reference, nuclei, 1, sample_key, 3, 5, mcmc_burnin_steps=2,
                mcmc_thinning_factor=1, **options,
            )
            samples = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), samples)
            optimize_wave_function_batched(
                reference, reference_opt, samples, nuclei, energy_key, batch_size=4
            )
        assert jnp.isclose(wf.alpha[...], reference.alpha[...], rtol=1e-6)
