import chex
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key
from flax import nnx
from .utils import Nucleus, Electron, clip_by_global_norm
from .electronic_optimization import WaveFunction, sample_from_wavefunction, hamiltonian
import einops
from functools import partial
import plotly.graph_objects as go


def nuclei_energy_grad_full(
    wave_function: WaveFunction,
    electron_samples: Electron,
    nuclei: Nucleus,
    key: Key,
    *,
    batch_size: int = 50,
    max_norm: float = 10.0,
) -> Array:
    """
    Compute the unbiased VMC nuclear energy gradient:
        grad_R <E> = E[grad_R E_L] + 2 * E[(E_L - <E_L>) * grad_R log|psi|]
    where the second term is the Pulay / score-function contribution arising from
    the R-dependence of the sampling distribution |psi(x; R)|^2.
    """
    chex.assert_shape(electron_samples.position, (None, None, 3))
    chex.assert_shape(nuclei.position, (None, 3))

    def hamiltonian_energy(
        nuclei_positions: Array,
        wf: WaveFunction,
        electron_sample: Electron,
        nuclei_charges: Array,
        sample_key: Key,
    ) -> Array:
        chex.assert_shape(electron_sample.position, (None, 3))
        chex.assert_shape(nuclei_positions, (None, 3))
        nuclei_obj = Nucleus(position=nuclei_positions, charge=nuclei_charges)
        energy, _details = hamiltonian(wf, electron_sample, nuclei_obj, sample_key)
        return energy

    def log_abs_psi(
        nuclei_positions: Array,
        wf: WaveFunction,
        electron_sample: Electron,
        nuclei_charges: Array,
    ) -> Array:
        nuclei_obj = Nucleus(position=nuclei_positions, charge=nuclei_charges)
        log_psi, _sign, _ortho_measure = wf(electron_sample, nuclei_obj)
        return log_psi

    energy_val_and_grad_fn = nnx.value_and_grad(hamiltonian_energy, argnums=0)
    log_psi_grad_fn = nnx.grad(log_abs_psi, argnums=0)

    def per_sample(
        wf: WaveFunction,
        electron_sample: Electron,
        nuclei_positions: Array,
        nuclei_charges: Array,
        sample_key: Key,
    ):
        local_energy, energy_grad = energy_val_and_grad_fn(
            nuclei_positions, wf, electron_sample, nuclei_charges, sample_key
        )
        energy_grad = clip_by_global_norm(energy_grad, max_norm=max_norm)

        log_psi_grad = log_psi_grad_fn(
            nuclei_positions, wf, electron_sample, nuclei_charges
        )
        log_psi_grad = clip_by_global_norm(log_psi_grad, max_norm=max_norm)

        weighted_log_psi_grad = local_energy * log_psi_grad
        return energy_grad, weighted_log_psi_grad, log_psi_grad, local_energy

    def sum_batch(
        wf: WaveFunction,
        batch_samples: Electron,
        nuclei_positions: Array,
        nuclei_charges: Array,
        batch_keys: Key,
    ):
        energy_grads, weighted_log_psi_grads, log_psi_grads, local_energies = nnx.vmap(
            per_sample, in_axes=(None, 0, None, None, 0)
        )(wf, batch_samples, nuclei_positions, nuclei_charges, batch_keys)
        return (
            jnp.sum(energy_grads, axis=0),
            jnp.sum(weighted_log_psi_grads, axis=0),
            jnp.sum(log_psi_grads, axis=0),
            jnp.sum(local_energies, axis=0),
        )

    @nnx.scan(in_axes=(None, nnx.Carry, 0, None, None, 0), out_axes=nnx.Carry)
    def scan_step(
        wf: WaveFunction,
        carry,
        batch_samples: Electron,
        nuclei_positions: Array,
        nuclei_charges: Array,
        batch_keys: Key,
    ):
        batch_sums = sum_batch(wf, batch_samples, nuclei_positions, nuclei_charges, batch_keys)
        return jax.tree.map(jnp.add, carry, batch_sums)

    num_samples = electron_samples.position.shape[0]
    hamiltonian_keys = jax.random.split(key, num_samples)

    effective_batch_size = min(batch_size, num_samples)
    num_full_batches = num_samples // effective_batch_size
    num_full_samples = num_full_batches * effective_batch_size

    init_grad = jnp.zeros_like(nuclei.position)
    init_carry = (init_grad, init_grad, init_grad, jnp.array(0.0, dtype=nuclei.position.dtype))

    full_samples = jax.tree.map(
        lambda x: x[:num_full_samples].reshape((num_full_batches, effective_batch_size) + x.shape[1:]),
        electron_samples,
    )
    full_keys = hamiltonian_keys[:num_full_samples].reshape(
        (num_full_batches, effective_batch_size) + hamiltonian_keys.shape[1:]
    )
    energy_grad_sum, weighted_log_psi_grad_sum, log_psi_grad_sum, local_energy_sum = scan_step(
        wave_function, init_carry, full_samples, nuclei.position, nuclei.charge, full_keys
    )

    if num_full_samples < num_samples:
        rem_samples = jax.tree.map(lambda x: x[num_full_samples:], electron_samples)
        rem_keys = hamiltonian_keys[num_full_samples:]
        rem_sums = sum_batch(wave_function, rem_samples, nuclei.position, nuclei.charge, rem_keys)
        energy_grad_sum, weighted_log_psi_grad_sum, log_psi_grad_sum, local_energy_sum = jax.tree.map(
            jnp.add,
            (energy_grad_sum, weighted_log_psi_grad_sum, log_psi_grad_sum, local_energy_sum),
            rem_sums,
        )

    mean_energy_grad = energy_grad_sum / num_samples
    mean_local_energy = local_energy_sum / num_samples
    mean_weighted_log_psi_grad = weighted_log_psi_grad_sum / num_samples
    mean_log_psi_grad = log_psi_grad_sum / num_samples

    # Use linearity of expectation: E[(E_L - <E_L>) * g] = E[E_L * g] - <E_L> * E[g]
    pulay_grad = 2.0 * (mean_weighted_log_psi_grad - mean_local_energy * mean_log_psi_grad)
    return mean_energy_grad + pulay_grad


def simplified_hamiltonian(electrons: Electron, nuclei: Nucleus):
    """
    For a given state |psi> computes the classical Coulomb potential energy
    (electron-nucleus attraction + nucleus-nucleus repulsion).
    """
    chex.assert_shape(electrons.position, (None, 3))
    chex.assert_shape(nuclei.position, (None, 3))

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


def nuclei_energy_grad_hellmann_feynman(
    _wave_function: WaveFunction,
    electron_samples: Electron,
    nuclei: Nucleus,
    key: Key,
    *,
    batch_size: int = 100,
    max_norm: float = 10.0,
) -> Array:
    """
    According to the Hellmann-Feynman theorem (https://en.wikipedia.org/wiki/Hellmann–Feynman_theorem),
    the force on the nuclei can be calculated only considering the classical columb forces between the nuclei and the electrons,
    and between the nuclei themselves.

    Notice, the wave_function itself is not used directly, but it is implicitly used to generate the electron_samples.
    It (along with the key) is kept in the function signature to be consistent with nuclei_energy_grad_full.
    """
    chex.assert_shape(electron_samples.position, (None, None, 3))
    chex.assert_shape(nuclei.position, (None, 3))

    def hamiltonian_energy(
        nuclei_positions: Array,
        electron_sample: Electron,
        nuclei_charges: Array,
    ):
        chex.assert_shape(electron_sample.position, (None, 3))
        chex.assert_shape(nuclei_positions, (None, 3))
        nuclei_obj = Nucleus(position=nuclei_positions, charge=nuclei_charges)
        return simplified_hamiltonian(electron_sample, nuclei_obj)

    energy_grad_fn = jax.grad(hamiltonian_energy, argnums=0)

    def per_sample(electron_sample: Electron) -> Array:
        grad = energy_grad_fn(nuclei.position, electron_sample, nuclei.charge)
        return clip_by_global_norm(grad, max_norm=max_norm)

    num_samples = electron_samples.position.shape[0]
    effective_batch_size = min(batch_size, num_samples)
    num_full_batches = num_samples // effective_batch_size
    num_full_samples = num_full_batches * effective_batch_size

    init_grad = jnp.zeros_like(nuclei.position)

    def scan_step(acc, batch_samples):
        batch_grads = jax.vmap(per_sample)(batch_samples)
        return acc + jnp.sum(batch_grads, axis=0), None

    full_samples = jax.tree.map(
        lambda x: x[:num_full_samples].reshape((num_full_batches, effective_batch_size) + x.shape[1:]),
        electron_samples,
    )
    grad_sum, _ = jax.lax.scan(scan_step, init_grad, full_samples)

    if num_full_samples < num_samples:
        rem_samples = jax.tree.map(lambda x: x[num_full_samples:], electron_samples)
        grad_sum = grad_sum + jnp.sum(jax.vmap(per_sample)(rem_samples), axis=0)

    return grad_sum / num_samples


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
    electron_samples = jax.tree.map(lambda x: einops.rearrange(x, "n c ... -> (n c) ..."), electron_samples)

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
