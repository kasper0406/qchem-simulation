"""Electronic VMC gradients and training."""

import chex
import einops
from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key
import optax

from .utils import Electron, Nucleus

# Re-export the model and physics helpers from their original import path.
from .hamiltonian import (
    HamiltonianParts,
    calculate_kinetic_energy,
    calculate_kinetic_energy_full,
    calculate_kinetic_energy_per_electron,
    hamiltonian,
)
from .sampling import (
    call_wavefunction_wrapper,
    init_electrons,
    per_electron_kernel,
    sample_from_wavefunction,
    walk_electrons,
)
from .wavefunction import (
    CuspElectronNuclei,
    CuspElectrons,
    Distance,
    DistanceEncoder,
    DistanceEncoders,
    Distances,
    ElectronicAttention,
    ExponentialEnvelope,
    FeedForward,
    FlatDistanceEncoder,
    JastrowFactor,
    OrbitalHead,
    OrbitalNetwork,
    SlaterDeterminant,
    StaticValue,
    TransformerLayer,
    TransformerStack,
    WaveFunction,
    calculate_distances,
    calculate_pairwise_distances,
    slater_log_sum_exp,
    slater_matrix_log_sum,
)


@nnx.jit
@nnx.value_and_grad(has_aux=True)
def log_prob_fn(wave_function: WaveFunction, sample: Electron, nuclei: Nucleus) -> Array:
    eps = 1e-12
    log_psi, _sign, _ortho_measure = wave_function(sample, nuclei)
    log_prob = 2 * log_psi
    return log_prob, log_psi

@nnx.jit
def local_energy(wave_function: WaveFunction, sample: Electron, nuclei: Nucleus, key: Key) -> Array:
    # print(f"Sample: {sample}")

    # Follows because we sample the samples according to the wave function
    # i.e. the integral we evaluate is already weighed by a |psi|^2
    # So we write:
    #   <L> = \int dx psi*(x) (H psi)(x) / \int dx psi*(x)psi(x)
    # WRITE:
    #   psi*(x) (H psi(x)) = (psi*(x) * psi(x)) * (H psi)(x)) / psi(x) (multiply and divide by psi(x))
    #                      = |psi(x)|^2 * Elocal(x)
    #
    # Notice that we do not need to normalize the wave function here, because
    # we are already sampling according to the wave function
    # Exactly because the norm factor is |psi|^2
    # norm_factor = jnp.conjugate(psi) * psi
    #
    # Notice that the Hamiltonian is already defined as H|psi> / psi
    # So we can just compute the local energy as:
    local_energy, hamiltonian_details = hamiltonian(wave_function, sample, nuclei, key)

    (log_prob, log_psi), log_grads = log_prob_fn(wave_function, sample, nuclei)
    # print(f"Log prob: {log_prob}")
    # print(f"Psi: {psi}")
    # print(f"Log grads: {log_grads}")

    return local_energy, log_grads, hamiltonian_details, log_psi


@nnx.jit
def optimize_wave_function_full(
    wave_function: nnx.Module,
    optimizer: nnx.Optimizer,
    samples: Electron,
    nuclei: Nucleus,
    key: Key,
    *,
    lambda_virial: float = 0.0,
) -> Array:
    """Perform a single VMC optimization step over all samples at once.

    Computes the unbiased VMC energy gradient:
        E[(E_L - <E_L>) * grad_theta log |psi|^2].

    Note: ``lambda_virial`` defaults to 0.0 because a pointwise (2*T_L + V)^2
    penalty is physically invalid (the virial theorem 2<T> + <V> = 0 holds only
    for expectation values at equilibrium nuclear geometry, and the score-function
    estimator omits the grad_theta T_L term).
    """
    keys = jax.random.split(key, samples.position.shape[0])

    local_energies, log_grads, hamiltonian_details, log_psi = nnx.vmap(local_energy, in_axes=(None, 0, None, 0))(wave_function, samples, nuclei, keys)

    avg_energy = jnp.mean(local_energies)

    virial_penalty = jnp.square(2 * hamiltonian_details.kinetic_energy + hamiltonian_details.potential_energy)
    penalty_terms = lambda_virial * virial_penalty

    loss_terms = local_energies + penalty_terms
    centered_loss_terms = loss_terms - jnp.mean(loss_terms)

    loss_gradient = jax.tree.map(
        lambda log_grad: jnp.expand_dims(centered_loss_terms, axis=tuple(range(1, log_grad.ndim))) * log_grad,
        log_grads,
    )

    mean_loss_grads = jax.tree.map(
        lambda lg: jnp.mean(lg, axis=0),
        loss_gradient,
    )

    optimizer.update(mean_loss_grads)

    mean_hamiltonian_details = jax.tree.map(lambda x: jnp.mean(x, axis=0), hamiltonian_details)
    return avg_energy, mean_hamiltonian_details, jnp.mean(log_psi)

@nnx.jit(static_argnames=["batch_size"])
def optimize_wave_function_batched(
    wave_function: WaveFunction,
    optimizer: nnx.Optimizer,
    samples: Electron,
    nuclei: Nucleus,
    key: Key,
    *,
    batch_size: int = 100,
    lambda_virial: float = 0.0,
) -> Array:
    """Perform a single VMC optimization step by accumulating gradients over batches.

    Computes the exact same VMC energy gradient as ``optimize_wave_function_full``:
        E[(E_L - <E_L>) * grad_theta log |psi|^2].

    Note: ``lambda_virial`` defaults to 0.0 because a pointwise (2*T_L + V)^2
    penalty is physically invalid (the virial theorem 2<T> + <V> = 0 holds only
    for expectation values at equilibrium nuclear geometry, and the score-function
    estimator omits the grad_theta T_L term).
    """

    @nnx.scan(
        in_axes=(None, nnx.Carry, 0, 0),
        out_axes=nnx.Carry,
    )
    def process_batch(wave_function: WaveFunction, carry, batch_samples: Electron, batch_key):
        weighed_grads_sum, local_energy_sum, loss_term_sum, grads_sum, hamiltonian_details_sum, log_psi_sum = carry
        keys = jax.random.split(batch_key, batch_samples.position.shape[0])

        local_energies, log_grads, hamiltonian_details, log_psi = nnx.vmap(local_energy, in_axes=(None, 0, None, 0))(wave_function, batch_samples, nuclei, keys)

        virial_penalty = jnp.square(2 * hamiltonian_details.kinetic_energy + hamiltonian_details.potential_energy)
        penalty_terms = lambda_virial * virial_penalty

        loss_terms = local_energies + penalty_terms

        weighed_grads_sum = jax.tree.map(
            lambda loss_gradient, acc: acc + jnp.sum(jnp.expand_dims(loss_terms, axis=tuple(range(1, loss_gradient.ndim))) * loss_gradient, axis=0),
            log_grads, weighed_grads_sum
        )

        grads_sum = jax.tree.map(
            lambda loss_gradient, acc: acc + jnp.sum(loss_gradient, axis=0),
            log_grads, grads_sum,
        )

        local_energy_sum = local_energy_sum + jnp.sum(local_energies, axis=0)
        loss_term_sum = loss_term_sum + jnp.sum(loss_terms, axis=0)
        hamiltonian_details_sum = jax.tree.map(lambda x, acc: acc + jnp.sum(x, axis=0), hamiltonian_details, hamiltonian_details_sum)
        log_psi_sum = log_psi_sum + jnp.sum(log_psi, axis=0)

        return weighed_grads_sum, local_energy_sum, loss_term_sum, grads_sum, hamiltonian_details_sum, log_psi_sum


    # Construct accumulators
    single_electron_samples = jax.tree.map(lambda x: x[0, ...], samples)
    _, grads_shape, hamiltonian_details_shape, _  = nnx.eval_shape(local_energy, wave_function, single_electron_samples, nuclei, jax.random.key(0))

    init_grads = jax.tree.map(jnp.zeros_like, grads_shape)
    init_weighed_grads = jax.tree.map(jnp.zeros_like, grads_shape)
    init_hamiltonian_details = jax.tree.map(jnp.zeros_like, hamiltonian_details_shape)

    # Re-arrange the input samples
    num_samples = samples.position.shape[0]
    chex.assert_is_divisible(num_samples, batch_size)
    num_batches = num_samples // batch_size
    batch_samples = jax.tree.map(lambda x: einops.rearrange(x, "(b n) ... -> b n ...", b=num_batches), samples)

    # Compute the batches sequentially
    batch_keys = jax.random.split(key, num_batches)
    weighed_grads_sum, local_energy_sum, loss_term_sum, grads_sum, hamiltonian_details_sum, log_psi_sum = process_batch(
        wave_function, (init_weighed_grads, 0.0, 0.0, init_grads, init_hamiltonian_details, 0.0), batch_samples, batch_keys
    )

    # Compute the mean loss grads by using linearity of expectations:
    #   E[(L - <L>) * log_grads] = E[L * log_grads] - <L> * E[log_grads]
    # where L = E_L + penalty_terms (with penalty_terms = 0 when lambda_virial = 0).
    # Then because we take the sum, we get the average by dividing by the number of samples.
    avg_loss_term = loss_term_sum / num_samples
    mean_loss_grads = jax.tree.map(
        lambda w, g: (w / num_samples) - avg_loss_term * (g / num_samples),
        weighed_grads_sum, grads_sum,
    )
    hamiltonian_details = jax.tree.map(lambda x: x / num_samples, hamiltonian_details_sum)
    log_psi = log_psi_sum / num_samples
    optimizer.update(mean_loss_grads)

    avg_energy = local_energy_sum / num_samples
    return avg_energy, hamiltonian_details, log_psi

optimize_wave_function = optimize_wave_function_batched


@chex.assert_max_traces(1)
@nnx.jit(static_argnames=["num_samples", "num_chains", "sum_of_charges"])
def sample_and_optimize_wave_function(
    wave_function: WaveFunction,
    optimizer: nnx.Optimizer,
    nuclei: Nucleus,
    sum_of_charges: int,
    num_chains: int,
    num_samples: int,
    key: Key
):
    chain_key, hamiltonian_key = jax.random.split(key, 2)
    samples, log_densities, acceptance_rate = sample_from_wavefunction(
        wave_function=wave_function,
        nuclei=nuclei,
        sum_of_charges=sum_of_charges,
        key=chain_key,
        num_chains=num_chains,
        num_samples=num_samples,
    )

    # Flatten the samples and log_densities for easier handlinga to be free of the chain dimension
    # print(f"Samples shape: {samples.position.shape}, Log densities shape: {log_densities.shape}")
    samples, log_densities = jax.tree.map(lambda x: einops.rearrange(x, "n c ... -> (n c) ..."), (samples, log_densities))

    avg_energy, hamiltonian_details, log_psi = optimize_wave_function(wave_function, optimizer, samples, nuclei, hamiltonian_key)

    return samples, avg_energy, hamiltonian_details, log_psi, acceptance_rate


def train_wavefunction(
    wave_function: WaveFunction,
    optimizer: nnx.Optimizer,
    nuclei: Nucleus,
    num_steps: int,
    num_samples: int,
    num_chains: int,
    key: Key,
):
    """
    Optimize the wave function by sampling and updating the parameters.
    """

    # Setup the MCMC sampler
    init_key, step_key = jax.random.split(key)
    sum_of_charges = int(jnp.sum(nuclei.charge))

    # Start the optimization loop
    cached_sample_and_optimize = nnx.cached_partial(
        sample_and_optimize_wave_function,
        wave_function,
        optimizer,
        nuclei,
        int(sum_of_charges),
        int(num_chains),
        int(num_samples),
    )

    step_keys = jax.random.split(step_key, num_steps)
    for step, step_key in enumerate(step_keys):
        print(f"Step {step}/{num_steps}")

        samples, avg_energy, hamiltonian_details, log_psi, acceptance_rate = cached_sample_and_optimize(step_key)

        print(f"Average energy step {step}: {avg_energy}")
        print(f"Hamiltonian details step {step}: {hamiltonian_details}")
        print(f"Log psi step {step}: {log_psi}")
        print(f"Acceptance rate {step}: {acceptance_rate}")

        energy_balance = 2 * hamiltonian_details.kinetic_energy + hamiltonian_details.potential_energy
        print(f"Constraints {step}: 2*T + V = {energy_balance}")


def main(args):
    # Setup nuclei
    # nuclei_positions = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=jnp.float32)
    # nuclei_charges = jnp.array([3, 1], dtype=jnp.int32)
    nuclei_positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    nuclei_charges = jnp.array([3], dtype=jnp.int32)
    nuclei = Nucleus(position=nuclei_positions, charge=nuclei_charges)

    # Initialize the wave function
    num_electrons = int(jnp.sum(nuclei.charge))
    num_nuclei = int(nuclei.position.shape[0])
    print(f"Number of electrons: {num_electrons}, Number of nuclei: {num_nuclei}")

    wave_function = WaveFunction(
        num_electrons=num_electrons,
        num_nuclei=num_nuclei,
        rngs=nnx.Rngs(0),
    )

    # Start the wave function optimization
    electronic_optimizer = nnx.ModelAndOptimizer(
        wave_function,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(1e-3),
        )
    )
    train_key = jax.random.key(0)

    total_steps = 1000
    num_chains = 50

    train_wavefunction(
        wave_function,
        electronic_optimizer,
        nuclei,
        num_steps=total_steps,
        num_samples=600,
        num_chains=num_chains,
        key=train_key
    )


if __name__ == "__main__":
    jax.config.update("jax_debug_nans", True)
    # jax.log_compiles(True)

    import os
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.90'
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_gemm=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
        '--xla_gpu_all_reduce_combine_threshold_bytes=51200 '
        '--xla_gpu_graph_level=0 '
        # '--xla_gpu_autotune_level=1 '
        '--xla_gpu_per_fusion_autotune_cache_dir=xla_autotune_results '

        '--xla_gpu_strict_conv_algorithm_picker=false '

        # '--xla_dump_to="/home/knielsen/xla_cuda_crash" '
    )

    from absl import app

    # with jax.profiler.trace("/home/knielsen/ml/models/qchem-simulation/tracing/jax-trace"):
    app.run(main)
