import chex
import jax
import jax.numpy as jnp
from flax import nnx

from qchem_simulation.electronic_optimization import (
    WaveFunction,
    init_electrons,
    per_electron_kernel,
    sample_from_wavefunction,
)
from qchem_simulation.utils import Electron, Nucleus


class GaussianWaveFunction(nnx.Module):
    """Analytical isotropic Gaussian wavefunction for testing MCMC sampling.

    Computes log_psi(r) = -0.5 * sum_i ||r_i - mu||^2 / sigma^2.
    Because MCMC samples from |psi|^2 = exp(2 * log_psi), each electron's
    stationary distribution is N(mu, (sigma^2 / 2) * I_3).
    """

    def __init__(self, mu: jax.Array, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, electrons: Electron, nuclei: Nucleus):
        diff = electrons.position - self.mu[None, :]
        log_psi = -0.5 * jnp.sum(diff ** 2) / (self.sigma ** 2)
        return log_psi, jnp.array(1.0), jnp.array(0.0)


def test_sample_from_wavefunction_acceptance_and_movement():
    """Verify sample_from_wavefunction has a healthy acceptance rate and moves electrons."""
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=jnp.float32),
        charge=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    sum_of_charges = 2
    num_chains = 4
    num_samples = 60
    burnin = 10
    thinning = 1

    wave_function = WaveFunction(num_electrons=sum_of_charges, num_nuclei=2, rngs=nnx.Rngs(0))
    key = jax.random.PRNGKey(42)

    samples, log_densities, acceptance_rate = sample_from_wavefunction(
        wave_function=wave_function,
        nuclei=nuclei,
        sum_of_charges=sum_of_charges,
        key=key,
        num_chains=num_chains,
        num_samples=num_samples,
        mcmc_burnin_steps=burnin,
        mcmc_thinning_factor=thinning,
    )

    expected_kept = (num_samples - burnin + thinning - 1) // thinning
    chex.assert_shape(samples.position, (expected_kept, num_chains, sum_of_charges, 3))
    chex.assert_shape(samples.spin, (expected_kept, num_chains, sum_of_charges, 1))
    assert jnp.all(jnp.isfinite(samples.position))
    assert jnp.all(jnp.isfinite(log_densities))
    assert jnp.isfinite(acceptance_rate)

    # MALA should have a healthy non-zero acceptance rate (with the init bug it was ~1e-30).
    assert float(acceptance_rate) > 0.1, f"Acceptance rate too low: {float(acceptance_rate)}"

    # Verify electrons actually moved from their initial positions and across sampling steps.
    init_electrons_key, _, _ = jax.random.split(key, 3)
    initial_electrons = init_electrons(
        nuclei, num_chains=num_chains, rng=init_electrons_key, sum_of_charges=sum_of_charges
    )
    displacement_from_init = jnp.linalg.norm(
        samples.position[-1] - initial_electrons.position, axis=-1
    )
    assert float(jnp.mean(displacement_from_init)) > 0.05

    step_displacements = jnp.linalg.norm(
        samples.position[1:] - samples.position[:-1], axis=-1
    )
    assert float(jnp.mean(step_displacements)) > 1e-3


def test_per_electron_kernel_analytical_gaussian():
    """Verify per_electron_kernel samples accurately from a known 3D isotropic Gaussian."""
    mu = jnp.array([1.2, -0.7, 0.5], dtype=jnp.float32)
    sigma = 1.0
    expected_var = (sigma ** 2) / 2.0  # 2 * log_psi => N(mu, sigma^2 / 2 * I)

    wave_function = GaussianWaveFunction(mu=mu, sigma=sigma)
    nuclei = Nucleus(
        position=jnp.zeros((1, 3), dtype=jnp.float32),
        charge=jnp.array([1.0], dtype=jnp.float32),
    )
    electron_spins = jnp.array([[1.0]], dtype=jnp.float32)
    num_electrons = 1
    num_chains = 16
    num_steps = 500
    burnin = 100

    def run_chain(chain_key):
        init_key, scan_key = jax.random.split(chain_key)
        init_pos = jax.random.normal(init_key, shape=(num_electrons, 3))
        step_keys = jax.random.split(scan_key, num_steps)

        def step_fn(pos, step_key):
            new_pos, log_density, acc_rate = per_electron_kernel(
                wave_function=wave_function,
                electron_spins=electron_spins,
                nuclei=nuclei,
                num_electrons=num_electrons,
                rng_key=step_key,
                electron_i=jnp.array(0, dtype=jnp.int32),
                electron_positions=pos,
            )
            return new_pos, (new_pos, log_density, acc_rate)

        _, (positions, log_densities, acc_rates) = jax.lax.scan(step_fn, init_pos, step_keys)
        return positions, log_densities, acc_rates

    chain_keys = jax.random.split(jax.random.PRNGKey(123), num_chains)
    positions, log_densities, acc_rates = jax.jit(jax.vmap(run_chain))(chain_keys)

    # Check returned log_density matches 2 * log_psi at the returned positions.
    expected_log_densities = -jnp.sum((positions[:, :, 0, :] - mu) ** 2, axis=-1) / (sigma ** 2)
    chex.assert_trees_all_close(log_densities, expected_log_densities, atol=1e-5)

    # Check acceptance rate is reasonable.
    mean_acc = float(jnp.mean(acc_rates[:, burnin:]))
    assert 0.3 < mean_acc < 1.0, f"Unexpected acceptance rate: {mean_acc}"

    # Discard burn-in and check empirical mean and variance against N(mu, (sigma^2/2) * I).
    samples = positions[:, burnin:, 0, :].reshape(-1, 3)
    empirical_mean = jnp.mean(samples, axis=0)
    empirical_var = jnp.var(samples, axis=0)

    chex.assert_trees_all_close(empirical_mean, mu, atol=0.1)
    chex.assert_trees_all_close(empirical_var, jnp.full((3,), expected_var), atol=0.1)


def test_sample_from_wavefunction_analytical_gaussian():
    """Verify full sample_from_wavefunction recovers an analytical 2-electron Gaussian."""
    mu = jnp.array([0.5, -0.5, 0.25], dtype=jnp.float32)
    sigma = 1.0
    expected_var = (sigma ** 2) / 2.0

    wave_function = GaussianWaveFunction(mu=mu, sigma=sigma)
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([2.0], dtype=jnp.float32),
    )

    samples, log_densities, acceptance_rate = sample_from_wavefunction(
        wave_function=wave_function,
        nuclei=nuclei,
        sum_of_charges=2,
        key=jax.random.PRNGKey(7),
        num_chains=16,
        num_samples=400,
        mcmc_burnin_steps=100,
        mcmc_thinning_factor=2,
    )

    assert 0.3 < float(acceptance_rate) < 1.0
    assert jnp.all(jnp.isfinite(log_densities))

    expected_log_densities = -jnp.sum(((samples.position - mu) / sigma) ** 2, axis=(-2, -1))
    chex.assert_trees_all_close(log_densities, expected_log_densities, atol=1e-5, rtol=1e-5)

    # Each electron independently follows N(mu, (sigma^2/2) * I).
    flat_positions = samples.position.reshape(-1, 3)
    empirical_mean = jnp.mean(flat_positions, axis=0)
    empirical_var = jnp.var(flat_positions, axis=0)

    chex.assert_trees_all_close(empirical_mean, mu, atol=0.1)
    chex.assert_trees_all_close(empirical_var, jnp.full((3,), expected_var), atol=0.1)


def test_sample_from_wavefunction_different_rng_keys():
    """Verify different RNG keys produce different samples and identical keys are deterministic."""
    wave_function = GaussianWaveFunction(
        mu=jnp.zeros(3, dtype=jnp.float32),
        sigma=1.0,
    )
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([2.0], dtype=jnp.float32),
    )

    kwargs = dict(
        wave_function=wave_function,
        nuclei=nuclei,
        sum_of_charges=2,
        num_chains=4,
        num_samples=30,
        mcmc_burnin_steps=5,
        mcmc_thinning_factor=1,
    )

    samples_a, log_dens_a, _ = sample_from_wavefunction(key=jax.random.PRNGKey(0), **kwargs)
    samples_b, log_dens_b, _ = sample_from_wavefunction(key=jax.random.PRNGKey(1), **kwargs)
    samples_a_repeat, log_dens_a_repeat, _ = sample_from_wavefunction(key=jax.random.PRNGKey(0), **kwargs)

    # Same key -> identical outputs.
    chex.assert_trees_all_close(samples_a.position, samples_a_repeat.position)
    chex.assert_trees_all_close(log_dens_a, log_dens_a_repeat)

    # Different keys -> different samples and log densities.
    assert not jnp.allclose(samples_a.position, samples_b.position)
    assert not jnp.allclose(log_dens_a, log_dens_b)
