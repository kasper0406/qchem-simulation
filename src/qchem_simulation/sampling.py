"""Initialization and MALA sampling for a fixed electronic wavefunction."""

from dataclasses import dataclass
from typing import Callable

import blackjax
import chex
from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .utils import Electron, Nucleus, clip_grad_global
from .wavefunction import WaveFunction, spin_up_count


@nnx.jit(static_argnames=["sum_of_charges", "num_chains", "num_up"])
def init_electrons(
    nuclei: Nucleus, num_chains: int, rng: Key, sum_of_charges: int, num_up: int | None = None
) -> Electron:
    """
    Initializes electrons around nuclei with random positions in a fixed spin sector.

    Args:
        nuclei: A list of Nucleus objects.
        num_chains: The number of simulation chains.
        rng: A JAX random key.
        num_up: Number of up spins. Defaults to half the electron count rounded up.

    Returns:
        An Electron object with initialized positions and spins.
    """
    # Repeat the nuclei positions according to their charges
    num_electrons = sum_of_charges
    positions = jnp.repeat(nuclei.position, nuclei.charge.astype(jnp.int32), axis=0, total_repeat_length=num_electrons)

    pos_rng, _ = jax.random.split(rng)

    # Generate random positions around the nuclei for each chain.
    # The positions are created with a shape of (num_chains, num_electrons, 3).
    sigma = 0.2  # Standard deviation for the normal distribution around each nucleus
    positions = jnp.expand_dims(positions, axis=0) + sigma * jax.random.normal(pos_rng, shape=(num_chains, num_electrons, 3))

    num_up = spin_up_count(num_electrons, num_up)
    num_down = num_electrons - num_up
    index = jnp.arange(num_electrons)
    # Alternate through the paired spins and append the unpaired majority.
    paired = index < 2 * min(num_up, num_down)
    is_up = jnp.where(paired, index % 2 == 0, num_up > num_down)
    spins = jnp.broadcast_to(jnp.where(is_up, 1.0, -1.0)[None, :, None], (num_chains, num_electrons, 1))
    return Electron(position=positions, spin=spins)


def walk_electrons(sigma: Array) -> Callable:
    sampler = blackjax.mcmc.random_walk.normal(sigma)

    def propose(rng_key: Key, positions: Array) -> Electron:
        new_positions = sampler(rng_key, positions)
        return new_positions

        # Flip the spin of the electrons with a probability of 0.1
        # flip_mask = jax.random.uniform(rng_key, shape=electrons.spin.shape) < 0.1
        # new_spins = jnp.where(flip_mask, -electrons.spin, electrons.spin)

        # return Electron(position=new_positions, spin=new_spins)

    return propose


# Code the actually do MCMC sampling
def call_wavefunction_wrapper(
    # The MCMC chain will move only the position of one electron at a time
    proposed_position_i: Array, # Shape (3,)
    electron_i: int,
    electron_positions: Array, # Shape (num_electrons, 3)
    *,
    electron_spins: Array,
    nuclei: Nucleus,
    wave_function: WaveFunction,
    num_electrons: int,
) -> Array:
    chex.assert_shape(proposed_position_i, (3,))
    chex.assert_shape(electron_positions, (num_electrons, 3))

    proposed_positions = electron_positions.at[electron_i].set(proposed_position_i)
    # print(f"Electrons shape: {all_positions.shape}")

    electrons = Electron(position=proposed_positions, spin=electron_spins)
    log_psi, _sign, _ortho_measure = wave_function(electrons, nuclei)
    # print(f"Wave function output shape: {log_psi.shape}")
    return 2 * log_psi

@jax.tree_util.register_dataclass
@dataclass
class WalkerState:
    """Persistent MCMC walker state carried across VMC optimization steps."""
    position: Array   # Shape (num_chains, num_electrons, 3)
    spin: Array       # Shape (num_chains, num_electrons, 1)
    step_size: Array  # Scalar float32 MALA step size


@nnx.jit(static_argnames=["sum_of_charges", "num_chains", "num_up"])
def init_walkers(
    nuclei: Nucleus,
    num_chains: int,
    rng: Key,
    sum_of_charges: int,
    init_step_size: float = 0.1,
    num_up: int | None = None,
) -> WalkerState:
    """Initialize persistent MCMC walkers around the nuclei."""
    electrons = init_electrons(
        nuclei=nuclei,
        num_chains=num_chains,
        rng=rng,
        sum_of_charges=sum_of_charges,
        num_up=num_up,
    )
    return WalkerState(
        position=electrons.position,
        spin=electrons.spin,
        step_size=jnp.asarray(init_step_size, dtype=jnp.float32),
    )


def adapt_step_size(
    step_size: ArrayLike,
    acceptance_rate: ArrayLike,
    target_acceptance: float = 0.5,
    adapt_rate: float = 0.05,
    min_step_size: float = 1e-4,
    max_step_size: float = 2.0,
) -> Array:
    """Multiplicative (log-space Robbins-Monro) step-size adaptation.

    Increases step size when acceptance_rate > target_acceptance and decreases
    it when acceptance_rate < target_acceptance, clamped to [min_step_size, max_step_size].
    When adapt_rate == 0.0, returns step_size unchanged.
    """
    step_size = jnp.asarray(step_size, dtype=jnp.float32)
    acceptance_rate = jnp.asarray(acceptance_rate, dtype=jnp.float32)
    updated = step_size * jnp.exp(adapt_rate * (acceptance_rate - target_acceptance))
    clamped = jnp.clip(updated, min_step_size, max_step_size)
    return jnp.where(adapt_rate > 0.0, clamped, step_size)


def per_electron_kernel(
    wave_function: WaveFunction,
    electron_spins: Array,
    nuclei: Nucleus,
    num_electrons: int,
    rng_key: Key,
    electron_i: Array, # Scalar, which electron to update
    electron_positions: Array, # Shape (num_electrons, 3)
    step_size: ArrayLike = 0.1,
):
    mala_kernel = blackjax.mala.build_kernel()
    step_size = jnp.asarray(step_size, dtype=jnp.float32)

    # The gradient of the wave-functions tends to infinity around the cusps and nodes.
    # Clip the gradient during sampling to make it more stable.
    @clip_grad_global(max_norm=1.0)
    def per_electron_logdensity(
        proposed_position_i: Array, # Shape (3,)
    ) -> Array:
        return call_wavefunction_wrapper(
            proposed_position_i=proposed_position_i,
            electron_i=electron_i,
            electron_positions=electron_positions,
            wave_function=wave_function,
            electron_spins=electron_spins,
            nuclei=nuclei,
            num_electrons=num_electrons,
        )
    chain_state = blackjax.mala.init(electron_positions[electron_i], logdensity_fn=per_electron_logdensity)

    chain_state, info = mala_kernel(rng_key, chain_state, per_electron_logdensity, step_size)

    acceptance_rate = info.acceptance_rate
    updated_electron_i_pos = chain_state.position
    log_density = chain_state.logdensity
    new_electron_positions = electron_positions.at[electron_i].set(updated_electron_i_pos)

    return new_electron_positions, log_density, acceptance_rate


@nnx.jit(static_argnames=["num_steps", "mcmc_burnin_steps", "mcmc_thinning_factor"])
def step_walkers(
    wave_function: WaveFunction,
    nuclei: Nucleus,
    walker_state: WalkerState,
    key: Key,
    num_steps: int,
    target_acceptance: float = 0.5,
    adapt_rate: float = 0.05,
    min_step_size: float = 1e-4,
    max_step_size: float = 2.0,
    mcmc_burnin_steps: int = 0,
    mcmc_thinning_factor: int = 1,
):
    """Advance persistent MCMC walkers by `num_steps` sweeps with optional adaptive step size.

    Each sweep updates every electron once in a random permutation order per chain using MALA.
    Only the first `mcmc_burnin_steps` discarded sweeps may adapt the step size.

    Returns:
        samples: Electron with position shape (num_kept_steps, num_chains, num_electrons, 3)
            and spin shape (num_kept_steps, num_chains, num_electrons, 1).
        new_walker_state: Updated WalkerState with final positions, spins, and step size.
        log_densities: Array of shape (num_kept_steps, num_chains) with log densities of the retained configurations.
        acceptance_rate: Scalar mean acceptance rate across all steps and chains.
    """
    num_chains, num_electrons, _ = walker_state.position.shape
    perm_key, step_key = jax.random.split(key, 2)

    # Update electron positions one by one to keep acceptance rates healthy.
    @nnx.scan(in_axes=(None, 0, nnx.Carry, None, 0, None), out_axes=(nnx.Carry, 0, 0))
    def scan_electrons(
        wave_function: WaveFunction,
        electron_i: Array, # Scalar, which electron to update
        electron_positions: Array, # Shape (num_electrons, 3)
        electron_spins: Array, # Shape (num_electrons, 1)
        electron_step_key: Key,
        step_size: Array,
    ):
        new_electron_positions, log_density, acceptance_rate = per_electron_kernel(
            wave_function=wave_function,
            electron_spins=electron_spins,
            nuclei=nuclei,
            num_electrons=num_electrons,
            rng_key=electron_step_key,
            electron_i=electron_i,
            electron_positions=electron_positions,
            step_size=step_size,
        )
        return new_electron_positions, log_density, acceptance_rate

    @nnx.vmap(in_axes=(None, 0, 0, 0, 0, None), out_axes=0)
    def map_chains(
        wave_function: WaveFunction,
        electron_indices: Array, # Shape (num_chains, num_electrons)
        electron_positions: Array, # Shape (num_chains, num_electrons, 3)
        electron_spins: Array, # Shape (num_chains, num_electrons, 1)
        chain_step_key: Key,
        step_size: Array,
    ):
        new_electron_positions, log_density, acceptance_rate = scan_electrons(
            wave_function,
            electron_indices,
            electron_positions,
            electron_spins,
            jax.random.split(chain_step_key, num_electrons),
            step_size,
        )
        return new_electron_positions, log_density[-1], jnp.mean(acceptance_rate)

    @nnx.scan(in_axes=(None, 0, nnx.Carry, 0, 0), out_axes=(0, nnx.Carry, 0, 0), length=num_steps)
    def scan_num_steps(
        wave_function: WaveFunction,
        electron_indices: Array, # Shape (num_steps, num_chains, num_electrons)
        carry: tuple[Array, Array], # (electron_positions, step_size)
        step_rng: Key,
        sweep_index: Array,
    ):
        electron_positions, current_step_size = carry
        chex.assert_shape(electron_indices, (num_chains, num_electrons))
        chex.assert_shape(electron_positions, (num_chains, num_electrons, 3))
        chain_keys = jax.random.split(step_rng, num_chains)
        new_electron_positions, log_density, acceptance_rate = map_chains(
            wave_function,
            electron_indices,
            electron_positions,
            walker_state.spin,
            chain_keys,
            current_step_size,
        )
        step_acceptance = jnp.mean(acceptance_rate)
        next_step_size = adapt_step_size(
            current_step_size,
            step_acceptance,
            target_acceptance=target_acceptance,
            adapt_rate=jnp.where(sweep_index < mcmc_burnin_steps, adapt_rate, 0.0),
            min_step_size=min_step_size,
            max_step_size=max_step_size,
        )
        return new_electron_positions, (new_electron_positions, next_step_size), log_density, acceptance_rate

    @nnx.vmap(axis_size=num_steps)
    def generate_electron_permutation(rng_key: Key) -> Array:
        @nnx.vmap(axis_size=num_chains)
        def _generate(rng_key: Key) -> Array:
            return jax.random.permutation(rng_key, jnp.arange(num_electrons))
        return _generate(jax.random.split(rng_key, num_chains))

    electron_permutations = generate_electron_permutation(jax.random.split(perm_key, num_steps))
    chex.assert_shape(electron_permutations, (num_steps, num_chains, num_electrons))

    step_keys = jax.random.split(step_key, num_steps)
    init_step_size = jnp.asarray(walker_state.step_size, dtype=jnp.float32)
    electron_position_samples, (last_position, last_step_size), log_densities, acceptance_rates = scan_num_steps(
        wave_function,
        electron_permutations,
        (walker_state.position, init_step_size),
        step_keys,
        jnp.arange(num_steps),
    )
    chex.assert_shape(electron_position_samples, (num_steps, num_chains, num_electrons, 3))

    kept_positions = electron_position_samples[mcmc_burnin_steps::mcmc_thinning_factor, ...]
    kept_log_densities = log_densities[mcmc_burnin_steps::mcmc_thinning_factor, ...]
    replicated_spins = jnp.tile(walker_state.spin, (kept_positions.shape[0], 1, 1, 1))
    chex.assert_shape(replicated_spins, (kept_positions.shape[0], num_chains, num_electrons, 1))

    new_walker_state = WalkerState(
        position=last_position,
        spin=walker_state.spin,
        step_size=last_step_size,
    )
    return Electron(position=kept_positions, spin=replicated_spins), new_walker_state, kept_log_densities, jnp.mean(acceptance_rates)


@nnx.jit(static_argnames=["sum_of_charges", "num_chains", "num_samples", "mcmc_burnin_steps", "mcmc_thinning_factor"])
def sample_from_wavefunction(
    wave_function: WaveFunction,
    nuclei: Nucleus,
    sum_of_charges: int,
    key: Key,
    num_chains: int,
    num_samples: int,
    mcmc_burnin_steps: int = 50,
    mcmc_thinning_factor: int = 2,
    init_step_size: float = 0.1,
    target_acceptance: float = 0.5,
    adapt_rate: float = 0.0,
):
    init_electrons_key, sample_key = jax.random.split(key, 2)

    walkers = init_walkers(
        nuclei=nuclei,
        num_chains=num_chains,
        rng=init_electrons_key,
        sum_of_charges=sum_of_charges,
        init_step_size=init_step_size,
        num_up=getattr(wave_function, "num_up", None),
    )
    chex.assert_shape(walkers.position, (num_chains, sum_of_charges, 3))

    samples, _new_walkers, log_densities, acceptance_rate = step_walkers(
        wave_function=wave_function,
        nuclei=nuclei,
        walker_state=walkers,
        key=sample_key,
        num_steps=num_samples,
        target_acceptance=target_acceptance,
        adapt_rate=adapt_rate,
        mcmc_burnin_steps=mcmc_burnin_steps,
        mcmc_thinning_factor=mcmc_thinning_factor,
    )
    return samples, log_densities, acceptance_rate
