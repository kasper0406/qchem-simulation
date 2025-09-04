import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key
from flax import nnx
from .utils import Nucleus, Electron, reduce_vmap, clip_grad_elementwise, clip_grad_global
from .electronic_optimization import WaveFunction, sample_from_wavefunction, hamiltonian
import einops
from functools import partial


def nuclei_energy_grad(wave_function: WaveFunction, electron_samples: Electron, nuclei: Nucleus, key: Key):
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
    return grad / electron_samples.position.shape[0]  # We reduce by sum, so to get the mean force divide by the number of samples


def jiggle_nuclei(nuclei_positions: Array, key: Key, scale: float = 1e-2):
    noise = jax.random.normal(key, nuclei_positions.shape) * scale
    return nuclei_positions + noise


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
    sample_key, grad_key, jiggle_key = jax.random.split(key, 3)

    electron_samples, _log_densities, _chain_state, _acceptance_rate = sample_from_wavefunction(
        wave_function=wave_function,
        nuclei=nuclei,
        sum_of_charges=sum_of_charges,
        key=sample_key,
        num_chains=num_electron_chains,
        num_samples=num_electron_samples,
    )
    electron_samples = jax.tree.map(lambda x: einops.rearrange(x, "n c ... -> (n c) ..."), electron_samples)

    nuclei_grads = nuclei_energy_grad(wave_function, electron_samples, nuclei, grad_key)

    updates, opt_state = optimizer.update(nuclei_grads, opt_state, nuclei.position)
    new_position = optax.apply_updates(nuclei.position, updates)

    new_position = jiggle_nuclei(new_position, jiggle_key)

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
