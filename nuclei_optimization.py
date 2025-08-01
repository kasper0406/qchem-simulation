import jax
import jax.numpy as jnp
import optax
from utils import Nucleus, Electron
from jaxtyping import Array, Key, ArrayLike
from flax import nnx
from electronic_optimization import WaveFunction, sample_from_wavefunction, init_electrons
import einops

@nnx.value_and_grad
def hamiltonian(nuclei_position: Array, nuclei_charge: Array, electron_samples: Array):
    """
    nuclei_position: Array of shape (num_nuclei, 3) representing the positions of the nuclei.
    nuclei_charge: Array of shape (num_nuclei,) representing the charges of the nuclei.
    electron_samples: Array of shape (num_samples, num_electrons, 3) representing the sampled electron positions.

    The nuclei positions and charges are split because the differentiation of the Hamiltonian
    is done with respect to the nuclei positions, while the charges are constant.

    Calculate the Hamiltonian for the nuclei given the electrons.
    Notice given this is treated as a electrostatics problem, there is no kinetic energy term,
    so the Hamiltonian is simply the potential energy due to the interaction between nuclei and electrons.

    The potential between the nuclei and electrons is approximated from mcmc samples of the electrons from the wave function.
    """

    eps = 1e-8  # Small value to avoid division by zero

    # Potential energy between nuclei and electrons
    flat_electrons = einops.rearrange(electron_samples, "s e d -> (s e) d")
    electron_nuclei_distances = jnp.linalg.norm(
        flat_electrons[:, None] - nuclei_position[None, :], axis=-1
    )
    nucleus_electron_interaction = -jnp.sum(nuclei_charge[None, :] / (electron_nuclei_distances + eps))
    # Scale the interaction by the number of electrons
    nuclei_electrons = nucleus_electron_interaction / electron_samples.shape[0]

    # Potential energy between nuclei
    nuclei_i, nuclei_j = jnp.triu_indices(nuclei_position.shape[0], k=1)
    nuclei_charge = nuclei_charge[nuclei_i] * nuclei_charge[nuclei_j]
    nuclei_distances = jnp.linalg.norm(nuclei_position[nuclei_i] - nuclei_position[nuclei_j], axis=-1)
    nuclei_nuclei = jnp.sum(nuclei_charge / (nuclei_distances + eps))

    return nuclei_electrons + nuclei_nuclei


@nnx.jit(static_argnames=["num_electron_samples", "optimizer"])
def sample_and_optimize(
    wave_function: WaveFunction,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    nuclei: Nucleus,
    initial_electrons: Electron,
    num_electron_samples: int,
    key: Key,
):
    electron_samples, _, _ = sample_from_wavefunction(wave_function, nuclei, initial_electrons, num_electron_samples, key)
    electron_samples = electron_samples.position

    burnin = 100
    skip_factor = 2
    electron_samples = electron_samples[burnin::skip_factor]
    electron_samples = einops.rearrange(electron_samples, "n c ... -> (n c) ...")

    _hamiltonian_value, grads = hamiltonian(nuclei.position, nuclei.charge, electron_samples)

    updates, opt_state = optimizer.update(grads, opt_state, nuclei.position)
    new_position = optax.apply_updates(nuclei.position, updates)
    new_nuclei = Nucleus(position=new_position, charge=nuclei.charge)
    return new_nuclei, opt_state


def optimize_nuclei(
    wave_function: WaveFunction,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    num_steps: int,
    nuclei: Nucleus,
    key: Key,
):
    num_chains = 150
    num_samples = 1000

    step_keys = jax.random.split(key, num_steps)
    for step, step_key in zip(range(num_steps), step_keys):
        print(f"Nuclei optimization step {step + 1}/{num_steps}")
        initial_electrons = init_electrons(nuclei, num_chains=num_chains, rng=step_key)
        nuclei, opt_state = sample_and_optimize(
            wave_function,
            optimizer,
            opt_state,
            nuclei,
            initial_electrons,
            num_samples,
            step_key
        )

    print(f"New nuclei positions: {nuclei.position}")
    return nuclei, opt_state
