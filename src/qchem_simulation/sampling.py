"""Initialization and MALA sampling for a fixed electronic wavefunction."""

from typing import Callable

import blackjax
import chex
from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

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

def per_electron_kernel(
    wave_function: WaveFunction,
    electron_spins: Array,
    nuclei: Nucleus,
    num_electrons: int,
    rng_key: Key,
    electron_i: Array, # Scalar, which electron to update
    electron_positions: Array, # Shape (num_electrons, 3)
):
    mala_kernel = blackjax.mala.build_kernel()
    step_size = 0.1  # TODO: Figure out a good step size adaptation scheme

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
):
    init_electrons_key, perm_key, step_key = jax.random.split(key, 3)

    initial_positions = init_electrons(
        nuclei, num_chains=num_chains, rng=init_electrons_key, sum_of_charges=sum_of_charges,
        num_up=getattr(wave_function, "num_up", None),
    )
    chex.assert_shape(initial_positions.position, (num_chains, sum_of_charges, 3))
    num_electrons = initial_positions.position.shape[1]

    # We want to update the electron positions one by one - proposing moves of all of them
    # at the same time, makes the acceptance rate very low and can cause weird behaviors
    @nnx.scan(in_axes=(None, 0, nnx.Carry, 0), out_axes=(nnx.Carry, 0, 0))
    def scan_electrons(
        wave_function: WaveFunction,
        electron_i: Array, # Scalar, which electron to update on which chain
        electron_positions: Array, # Shape (num_electrons, 3): current positions of all electrons on this chain
        electron_step_key: Key,
    ):
        new_electron_positions, log_density, acceptance_rate = per_electron_kernel(
            wave_function=wave_function,
            electron_spins=initial_positions.spin[0],  # All chains have the same spins
            nuclei=nuclei,
            num_electrons=num_electrons,
            rng_key=electron_step_key,
            electron_i=electron_i,
            electron_positions=electron_positions,
        )
        # print(f"New chain state: {new_chain_state}")
        # electron_state_sample, log_density = new_chain_state

        return new_electron_positions, log_density, acceptance_rate

    @nnx.vmap(in_axes=(None, 0, 0, 0), out_axes=0)
    def map_chains(
        wave_function: WaveFunction,
        electron_indices: Array, # Shape (num_chains, num_electrons)
        electron_positions: Array, # Shape (num_chains, num_electrons, 3)
        step_key: Key,
    ):
        new_electron_positions, log_density, acceptance_rate = scan_electrons(
            wave_function,
            electron_indices,
            electron_positions,
            jax.random.split(step_key, num_electrons),
        )

        return new_electron_positions, log_density[-1], jnp.mean(acceptance_rate)

    @nnx.scan(in_axes=(None, 0, nnx.Carry, 0), out_axes=(0, nnx.Carry, 0, 0), length=num_samples)
    def scan_num_samples(
        wave_function: WaveFunction,
        electron_indices: Array, # Shape (num_samples, num_chains, num_electrons) the order in which to update the electrons
        electron_positions: Array, # Carry of shape (num_chains, num_electrons, 3)
        step_key: Key
    ):
        chex.assert_shape(electron_indices, (num_chains, num_electrons))
        chex.assert_shape(electron_positions, (num_chains, num_electrons, 3))
        chain_keys = jax.random.split(step_key, num_chains)
        new_electron_positions, log_density, acceptance_rate = map_chains(
            wave_function,
            electron_indices,
            electron_positions,
            chain_keys,
        )

        return new_electron_positions, new_electron_positions, log_density, acceptance_rate

    # We want to update each electron in a turn, but we do not always want to update in the order 0, 1, 2, ...
    # We will generate electron_permutations of shape (num_samples, num_chains, num_electrons) where there will be a
    # random permutation of the electrons for each sample step and chain
    # TODO: Refactor this to a utility function?
    @nnx.vmap(axis_size=num_samples)
    def generate_electron_permutation(key: Key) -> Array:
        @nnx.vmap(axis_size=num_chains)
        def _generate(key: Key) -> Array:
            return jax.random.permutation(key, jnp.arange(num_electrons))
        return _generate(jax.random.split(key, num_chains))
    electron_permutations = generate_electron_permutation(jax.random.split(perm_key, num_samples))
    chex.assert_shape(electron_permutations, (num_samples, num_chains, num_electrons))

    step_keys = jax.random.split(step_key, num_samples)
    electron_position_samples, _last_position, log_densities, acceptance_rate = scan_num_samples(
        wave_function,
        electron_permutations,
        initial_positions.position,
        step_keys
    )
    chex.assert_shape(electron_position_samples, (num_samples, num_chains, num_electrons, 3))
    # print(f"Electron position shape: {electron_position_samples.shape}, initial position spin shape: {initial_positions.spin.shape}")
    # replicated_spins = jnp.tile(initial_positions.spin, (num_samples, 1, 1, 1))

    # Apply burn-in and thinning
    kept_steps = slice(mcmc_burnin_steps, None, mcmc_thinning_factor)
    electron_position_samples = electron_position_samples[kept_steps]
    log_densities = log_densities[kept_steps]
    replicated_spins = jnp.tile(initial_positions.spin, (electron_position_samples.shape[0], 1, 1, 1))

    chex.assert_shape(replicated_spins, (electron_position_samples.shape[0], num_chains, num_electrons, 1))
    return Electron(position=electron_position_samples, spin=replicated_spins), log_densities, jnp.mean(acceptance_rate)
