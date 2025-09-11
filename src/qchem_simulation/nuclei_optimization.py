import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key
from flax import nnx
from .utils import Nucleus, Electron, reduce_vmap, clip_grad_elementwise, clip_grad_global
from .electronic_optimization import WaveFunction, sample_from_wavefunction, hamiltonian
import einops
from functools import partial
import plotly.graph_objects as go


def nuclei_energy_grad_full(wave_function: WaveFunction, electron_samples: Electron, nuclei: Nucleus, key: Key):
    # @clip_grad_elementwise(-1.0, 1.0)
    @clip_grad_global(max_norm=10.0)
    def hamiltonian_grad(
        nuclei_positions: Array,  # Gradient wrt. nuclei positions
        wave_func_graphdef: WaveFunction,
        wave_func_state: WaveFunction,
        electron_sample: Electron,
        nuclei_charges: Array,
        key: Key
    ):
        wave_function = nnx.merge(wave_func_graphdef, wave_func_state)
        nuclei = Nucleus(position=nuclei_positions, charge=nuclei_charges)
        energy, _details = hamiltonian(wave_function, electron_sample, nuclei, key)
        return energy

    zero_grads = jnp.zeros_like(nuclei.position)
    @reduce_vmap(in_axes=(None, 0, None, None, 0), out_axes=0, init=zero_grads, reduce_fn=jnp.sum, acc_fn=jnp.add, batch_size=50)
    def energy_grad(
        wave_function: WaveFunction,
        electron_sample: Electron,
        nuclei_positions: Array,
        nuclei_charges: Array,
        key: Key
    ) -> Array:
        wave_func_graphdef, wave_func_state = nnx.split(wave_function)
        energy_grad_fn = jax.grad(hamiltonian_grad)
        # print(f"Nuclei positions: {nuclei_positions}")
        energy_grad = energy_grad_fn(nuclei_positions, wave_func_graphdef, wave_func_state, electron_sample, nuclei_charges, key)
        return energy_grad

    hamiltonian_keys = jax.random.split(key, electron_samples.position.shape[0])
    # Differentiate wrt nuclei_positions
    grad = energy_grad(wave_function, electron_samples, nuclei.position, nuclei.charge, hamiltonian_keys)
    return grad / electron_samples.position.shape[0]  # We reduce by sum, so to get the mean energy gradient divide by the number of samples


def nuclei_energy_grad_hellmann_feynman(_wave_function: WaveFunction, electron_samples: Electron, nuclei: Nucleus, key: Key):
    """
    According to the Hellmann-Feynman theorem (https://en.wikipedia.org/wiki/Hellmann–Feynman_theorem),
    the force on the nuclei can be calculated only considering the classical columb forces between the nuclei and the electrons,
    and between the nuclei themselves.

    Notice, the wave_function itself is not used directly, but it is implicitly used to generate the electron_samples.
    It (along with the key) is kept in the function signature to be consistent with nuclei_energy_grad_full.
    """
    def simplified_hamiltonian(electrons: Electron, nuclei: Nucleus):
        """
        For a given state |psi> computes: H|psi> / psi (the local energy).
        """
        eps = 1e-12

        nuclea_i, nuclea_j = jnp.triu_indices(nuclei.position.shape[0], k=1)
        nuclei_charge = nuclei.charge[nuclea_i] * nuclei.charge[nuclea_j]
        nuclei_distances = jnp.linalg.norm(nuclei.position[nuclea_i] - nuclei.position[nuclea_j], axis=-1)
        nuclea_interaction = jnp.sum(nuclei_charge / (nuclei_distances + eps))

        electron_nuclei_distances = jnp.linalg.norm(
            electrons.position[:, None] - nuclei.position[None, :], axis=-1
        )
        nucleus_electron_interaction = -jnp.sum(nuclei.charge[None, :] / (electron_nuclei_distances + eps))

        return nucleus_electron_interaction + nuclea_interaction

    @clip_grad_global(max_norm=10.0)
    def hamiltonian_grad(
        nuclei_positions: Array,  # Gradient wrt. nuclei positions
        electron_sample: Electron,
        nuclei_charges: Array,
    ):
        nuclei = Nucleus(position=nuclei_positions, charge=nuclei_charges)
        energy= simplified_hamiltonian(electron_sample, nuclei)
        return energy

    zero_grads = jnp.zeros_like(nuclei.position)
    @reduce_vmap(in_axes=(0, None, None), out_axes=0, init=zero_grads, reduce_fn=jnp.sum, acc_fn=jnp.add, batch_size=100)
    def energy_grad(
        electron_sample: Electron,
        nuclei_positions: Array,
        nuclei_charges: Array,
    ) -> Array:
        energy_grad_fn = jax.grad(hamiltonian_grad)
        print(f"Nuclei positions: {nuclei_positions}")
        print(f"Nuclei charges: {nuclei_charges}")
        print(f"Electron samples: {electron_sample}")
        energy_grad = energy_grad_fn(nuclei_positions, electron_sample, nuclei_charges)
        return energy_grad

    # Differentiate wrt nuclei_positions
    grad = energy_grad(electron_samples, nuclei.position, nuclei.charge)
    return grad / electron_samples.position.shape[0]  # We reduce by sum, so to get the mean energy gradient divide by the number of samples


@nnx.jit(static_argnames=["num_electron_samples", "num_electron_chains", "sum_of_charges", "optimizer"])
def sample_and_optimize(
    wave_function: WaveFunction,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    nuclei: Nucleus,
    num_electron_samples: int,
    num_electron_chains: int,
    sum_of_charges: int,
    key: Key,
):
    sample_key, grad_key = jax.random.split(key, 2)

    electron_samples, _log_densities, _acceptance_rate = sample_from_wavefunction(
        wave_function=wave_function,
        nuclei=nuclei,
        sum_of_charges=sum_of_charges,
        key=sample_key,
        num_chains=num_electron_chains,
        num_samples=num_electron_samples,
    )
    electron_samples = jax.tree.map(lambda x: einops.rearrange(x, "n c s ... -> (n c s) ..."), electron_samples)

    nuclei_grads = nuclei_energy_grad_hellmann_feynman(wave_function, electron_samples, nuclei, grad_key)

    updates, opt_state = optimizer.update(nuclei_grads, opt_state, nuclei.position)
    new_position = optax.apply_updates(nuclei.position, updates)

    new_nuclei = Nucleus(position=new_position, charge=nuclei.charge)

    return new_nuclei, opt_state


def optimize_nuclei(
    wave_function: WaveFunction,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    num_steps: int,
    nuclei: Nucleus,
    num_electron_samples: int,
    num_electron_chains: int,
    key: Key,
):
    sum_of_charges = int(jnp.sum(nuclei.charge))
    step_keys = jax.random.split(key, num_steps)
    for step, step_key in zip(range(num_steps), step_keys):
        print(f"Nuclei optimization step {step + 1}/{num_steps}")

        nuclei, opt_state = sample_and_optimize(
            wave_function,
            optimizer,
            opt_state,
            nuclei,
            num_electron_samples=num_electron_samples,
            num_electron_chains=num_electron_chains,
            sum_of_charges=sum_of_charges,
            key=step_key,
        )

        print(f"Nuclei positions {step + 1}: {nuclei.position}")

    nuclei_i, nuclei_j = jnp.triu_indices(nuclei.position.shape[0], k=1)
    nuclei_distances = jnp.linalg.norm(nuclei.position[nuclei_i] - nuclei.position[nuclei_j], axis=-1)
    print(f"Nuclei distances: {nuclei_distances}")

    return nuclei, opt_state
