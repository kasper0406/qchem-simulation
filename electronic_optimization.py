import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from jaxtyping import Array, Key, ArrayLike
from typing import List, Optional, Callable
from folx import forward_laplacian
import blackjax
from flax import nnx
import einops
from functools import partial
import plotly.graph_objects as go
import optax
from utils import Electron, Nucleus

@jax.tree_util.register_dataclass
@dataclass
class HamiltonianParts:
    kinetic_energy: Array
    potential_energy: Array

    # Partial contributions to the potential energy
    electron_repulsion: Array
    nucleus_electron_interaction: Array
    nucleus_nucleus_interaction: Array

@jax.tree_util.register_dataclass
@dataclass
class Distances:
    electron_distances: Array
    nuclei_distances: Array
    electron_nuclei_distances: Array


# @nnx.jit
def calculate_kinetic_energy(wave_function: "WaveFunction", electrons: Electron, nuclei: Nucleus) -> Array:
    position_indices = jnp.arange(electrons.position.shape[0])

    @nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
    def calc_terms(wave_function: WaveFunction, position: Array, electron_idx: int) -> Array:
        # Evaluate the wave function with respect to the electron's position
        def calc_wrt_position(position: Array, graphdef: nnx.GraphDef, state: nnx.State) -> Array:
            wave_function = nnx.merge(graphdef, state)
            amplitude, _sign = wave_function.eval_wrt_electron_position(position, nuclei, electron_idx, electrons.position, electrons.spin)
            return amplitude

        graphdef, state = nnx.split(wave_function)
        laplacian = forward_laplacian(partial(calc_wrt_position, graphdef=graphdef, state=state))(position)
        # print(f"Jacobian: {amplitude_laplacian.jacobian}")
        # print(f"Laplacian: {amplitude_laplacian}")

        # Use the identity: ∇^2 psi = psi * (∇^2 log(psi) + |∇log(psi)|^2)
        # We actually want to calculate (∇^2 psi)/psi (the local energy, and from linearity of the Laplacian term
        # in the kinetic energy). Therefore, we calculate: (∇^2 psi)/psi = (∇^2 log(psi) + |∇log(psi)|^2)
        # where |∇log(psi)|^2 = J \dot J^T where J is the Jacobian of the wave function
        # Notice that since the laplacian is linear, the sign of psi will cancel out in (∇^2 psi)/psi.
        amplitude = laplacian.laplacian + jnp.dot(laplacian.dense_jacobian, laplacian.dense_jacobian)
        return amplitude

    terms = calc_terms(wave_function, electrons.position, position_indices)
    # print(f"Laplacian terms shape: {terms.shape}")
    return -0.5 * jnp.sum(terms)

# @nnx.jit
def hamiltonian(wave_function: "WaveFunction", electrons: Electron, nuclei: Nucleus):
    """
    For a given state |psi> computes: H|psi> / psi (the local energy).
    """
    eps = 1e-8

    nuclea_i, nuclea_j = jnp.triu_indices(nuclei.position.shape[0], k=1)
    nuclei_charge = nuclei.charge[nuclea_i] * nuclei.charge[nuclea_j]
    nuclei_distances = jnp.linalg.norm((nuclei.position[nuclea_i] - nuclei.position[nuclea_j]) + eps, axis=-1)
    nuclea_interaction = jnp.sum(nuclei_charge / (nuclei_distances + eps))

    electron_repulsion = 0.0
    if electrons.position.shape[0] > 1:
        electron_i, electron_j = jnp.triu_indices(electrons.position.shape[0], k=1)
        electron_distances = jnp.linalg.norm((electrons.position[electron_i] - electrons.position[electron_j]) + eps, axis=-1)
        electron_repulsion = jnp.sum(1.0 / electron_distances)

    electron_nuclei_distances = jnp.linalg.norm(
        (electrons.position[:, None] - nuclei.position[None, :]) + eps, axis=-1
    )
    nucleus_electron_interaction = -jnp.sum(nuclei.charge[None, :] / (electron_nuclei_distances + eps))

    # The wave_function returns the log-probability, and the calculated kinetic energy is
    # already normalized by psi. See the comments in the `calculate_kinetic_energy` function.
    kinetic_energy = calculate_kinetic_energy(wave_function, electrons, nuclei)
    potential_energy = electron_repulsion + nucleus_electron_interaction + nuclea_interaction

    result = kinetic_energy + potential_energy
    return result, HamiltonianParts(
        kinetic_energy=kinetic_energy,
        potential_energy=potential_energy,

        electron_repulsion=electron_repulsion,
        nucleus_electron_interaction=nucleus_electron_interaction,
        nucleus_nucleus_interaction=nuclea_interaction,
    )


class FeedForward(nnx.Module):
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        self.up_project = nnx.Linear(hidden_dim, hidden_dim * 4, rngs=rngs)
        self.down_project = nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = self.up_project(x)
        x = nnx.glu(x)
        return self.down_project(x)


class TransformerLayer(nnx.Module):
    def __init__(self, num_heads: int, hidden_dim: int, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            decode=False,
            rngs=rngs
        )
        self.feed_forward = FeedForward(hidden_dim, rngs)
        
        self.pre_attn_norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.post_attn_norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)

    def __call__(self, key_values: Array, query: Array | None = None) -> Array:
        if query is None:
            query = key_values

        h = self.pre_attn_norm(key_values)
        r = self.attention(
            inputs_q=query,
            inputs_k=h,
            inputs_v=h,
        )
        h = key_values + r

        normalized_h = self.post_attn_norm(h)
        r = nnx.vmap(self.feed_forward)(normalized_h)
        return h + r


class TransformerStack(nnx.Module):
    def __init__(self, num_layers: int, num_heads: int, hidden_dim: int, rngs: nnx.Rngs):
        # @nnx.split_rngs(splits=num_layers)
        # @nnx.vmap
        # def create_layer(rng: Key) -> TransformerLayer:
        #     return TransformerLayer(num_heads=num_heads, hidden_dim=hidden_dim, rngs=rng)

        # self.layers = create_layer(rngs)
        self.layers = []
        for _ in range(num_layers):
            layer = TransformerLayer(num_heads=num_heads, hidden_dim=hidden_dim, rngs=rngs)
            self.layers.append(layer)

    def __call__(self, key_values: Array, query: Array | None = None) -> Array:
        # @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        # def forward(layer: TransformerLayer, x: Array) -> Array:
        #     return layer(x)

        # return forward(self.layers, inputs)
        h = key_values
        for layer in self.layers:
            h = layer(key_values, query)
        return h



def calculate_pairwise_distances(position_a: Array, position_b: Array, upper_half: bool = False) -> Array:
    """
    Calculate distance features between two sets of positions.
    Returns the squared distances and the inverse distances.
    """
    eps = 1e-6
    diff = position_a[:, None] - position_b[None, :]
    square_distances = jnp.sum(jnp.square(diff), axis=-1)
    distances = jnp.sqrt(square_distances + eps)

    if upper_half:
        # Keep only the upper half of the matrix (i.e., distances between different electrons)
        print("Inside upper half calculation")
        upper_indices = jnp.triu_indices(distances.shape[0], k=1)
        distances = jnp.reshape(distances[upper_indices], (-1, ))
        square_distances = jnp.reshape(square_distances[upper_indices], (-1, ))

    return distances, square_distances


def calculate_distances(electron_positions: Array, nucleus_positions: Array) -> Distances:
    """
    Calculate distances between electrons and nuclei.
    Returns a Distances object containing:
        - electron_distances: Distances between electrons
        - nuclei_distances: Distances between nuclei
        - electron_nuclei_distances: Distances between electrons and nuclei
    """
    electron_distances, _ = calculate_pairwise_distances(electron_positions, electron_positions)
    nucleus_distances, _ = calculate_pairwise_distances(nucleus_positions, nucleus_positions, upper_half=True)
    electron_nuclei_distances, _ = calculate_pairwise_distances(electron_positions, nucleus_positions)

    return Distances(
        electron_distances=electron_distances,
        nuclei_distances=nucleus_distances,
        electron_nuclei_distances=electron_nuclei_distances,
    )


@nnx.vmap(in_axes=(0, None), out_axes=0)
def calculate_distance_tokens(position_a: Array, position_b: Array) -> Array:
    square_distances = jnp.sum((position_a[None, :] - position_b) ** 2, axis=-1)
    distances = jnp.sqrt(square_distances)
    return square_distances, distances


class ElectronicAttention(nnx.Module):
    def __init__(self, hdim: int, num_electrons: int, num_nuclei: int, rngs: nnx.Rngs):
        self.electron_distance_encoder = nnx.Linear(num_electrons, hdim, rngs=rngs)
        self.electron_nuclei_distance_encoder = nnx.Linear(num_nuclei, hdim, rngs=rngs)

        # Set up the nuclei-nuclei distance encoder
        self.nuclei_distance_encoder = None
        nuclei_pairs = num_nuclei * (num_nuclei - 1) // 2
        if nuclei_pairs > 0:
            self.nuclei_distance_encoder = nnx.Linear(nuclei_pairs, hdim, rngs=rngs)

        # TODO: Implement proper attention
        self.transformer = TransformerStack(
            num_layers=4,
            num_heads=1,
            hidden_dim=hdim,
            rngs=rngs,
        )
        self.down_project = nnx.Linear(hdim, 1, rngs=rngs)

    def __call__(self, distances: Distances, extra_features: Array | None = None) -> Array:
        print("Electron distances:", distances.electron_distances.shape)
        print("Nuclei distances:", distances.nuclei_distances.shape)
        print("Electron-nuclei distances:", distances.electron_nuclei_distances.shape)
        electron_distance_embedding = self.electron_distance_encoder(distances.electron_distances)
        electron_nuclei_distance_embedding = self.electron_nuclei_distance_encoder(distances.electron_nuclei_distances)
        distance_embedding = electron_distance_embedding + electron_nuclei_distance_embedding
        
        if self.nuclei_distance_encoder is not None:
            nucleus_distance_embedding = self.nuclei_distance_encoder(distances.nuclei_distances)
            distance_embedding = distance_embedding + nucleus_distance_embedding[None, :]

        combined_embedding = distance_embedding
        if extra_features is not None:
            combined_embedding += extra_features

        print(f"Combined embedding shape: {combined_embedding.shape}")
        attention_output = self.transformer(combined_embedding)
        # attention_output = self.transformer(combined_embedding, electron_distance_embedding)

        return self.down_project(attention_output)


# The following constructs a wave function using a neural network, following the
# terminology described by https://en.wikipedia.org/wiki/Slater_determinant#Multi-particle_case
class Chi(nnx.Module):
    def __init__(self, hdim: int, num_electrons: int, num_nuclei: int, rngs: nnx.Rngs):
        self.spin_encoder = nnx.Embed(2, features=hdim, rngs=rngs)
        self.electronic_attention = ElectronicAttention(hdim, num_electrons, num_nuclei, rngs)

    def __call__(self, distances: Distances, spins: Array) -> Array:
        spin_classes = jnp.squeeze(jnp.where(spins > 0, 1, 0))
        # print("Spin classes:", spin_classes)
        spin_embedding = self.spin_encoder(spin_classes)

        attention = self.electronic_attention(distances, extra_features=spin_embedding)
        return attention


class SlaterDeterminant(nnx.Module):
    def __init__(self, num_electrons: int, num_nuclei: int, hidden_dim: int, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=num_electrons)
        @nnx.vmap(in_axes=(None, 0))
        def create_chis(num_electrons: int, rng: Key) -> Chi:
            return Chi(hidden_dim, num_electrons, num_nuclei, rng)
        self.chis = create_chis(num_electrons, rngs)

    def __call__(self, distances: Array, spins: Array) -> Array:
        @nnx.vmap(in_axes=(0, None, None), out_axes=0)
        def calc_chis(chi: Chi, distances: Distances, spins: Array) -> Array:
            return chi(distances, spins)

        chi_matrix = calc_chis(self.chis, distances, spins)
        chi_matrix = jnp.squeeze(chi_matrix, axis=-1)

        print(f"Chi matrix shape: {chi_matrix.shape}")
        log_det = jnp.linalg.slogdet(chi_matrix)  # log determinant
        # print(f"Log-Determinant value: {log_det}")
        return log_det.logabsdet, log_det.sign


class JastrowFactor(nnx.Module):
    """
        Map from N * R^3 to R, where N is the number of electrons and R^3 is the position of each electron.
    """
    def __init__(self, hdim: int, num_electrons: int, num_nuclei: int, rngs: nnx.Rngs):
        self.electronic_attention = ElectronicAttention(hdim, num_electrons, num_nuclei, rngs)

    def __call__(self, distances: Array) -> Array:
        attention = self.electronic_attention(distances)
        return jnp.sum(attention)


@jax.jit
def slater_log_sum_exp(slater_logs, slater_signs, slater_coeffs):
    """
    Calculates the log-magnitude and sign of a wavefunction represented
    as a linear combination of Slater determinants.

    This function uses the log-sum-exp trick adapted for signed sums to
    ensure numerical stability.

    Args:
        slater_logs: An array of the log-absolute-values of the determinants.
        slater_signs: An array of the signs (+1 or -1) of the determinants.
        slater_coeffs: An array of the linear combination coefficients (c_i).

    Returns:
        A tuple containing:
        - log_psi_abs: The log of the absolute value of the total wavefunction.
        - sign_psi: The sign (+1 or -1) of the total wavefunction.
    """
    # Calculate the log-magnitude of each term in the expansion
    # log|T_i| = log|c_i| + log|D_i|
    log_term_magnitudes = jnp.log(jnp.abs(slater_coeffs)) + slater_logs

    # Find the dominant term's magnitude (the 'x_max' in the standard trick)
    max_log_mag = jnp.max(log_term_magnitudes)

    # Calculate the sum of terms, normalized by the largest term to prevent overflow.
    # The value of each term in the sum is:
    # sign(c_i)*s_i*exp(log|c_i| + ld_i - max_log_mag)
    sum_val = jnp.sum(
        jnp.sign(slater_coeffs) * slater_signs * jnp.exp(log_term_magnitudes - max_log_mag)
    )

    # The final sign is the sign of this normalized sum
    sign_psi = jnp.sign(sum_val)

    # The final log-magnitude is the log of the dominant term's magnitude
    # plus the log of the absolute value of the normalized sum.
    log_psi_abs = max_log_mag + jnp.log(jnp.abs(sum_val))

    return log_psi_abs, sign_psi


class WaveFunction(nnx.Module):
    def __init__(self, num_electrons: int, num_nuclei: int, rngs: nnx.Rngs):
        hidden_dim = 32
        self.jastrow_factor = JastrowFactor(num_electrons=num_electrons, num_nuclei=num_nuclei, hdim=hidden_dim, rngs=rngs)

        num_slaters = 1
        @nnx.split_rngs(splits=num_slaters)
        @nnx.vmap(in_axes=(None, None, 0))
        def create_slater_determinants(num_electrons: int, num_nuclei: int, rng: Key) -> SlaterDeterminant:
            return SlaterDeterminant(num_electrons, num_nuclei, hidden_dim, rng)
        self.slater_determinant = create_slater_determinants(num_electrons, num_nuclei, rngs)

        self.slater_strengths = nnx.Param(jnp.ones((num_slaters,), dtype=jnp.float32))

    def __call__(self, electrons: Electron, nuclei: Nucleus) -> Array:
        distances = calculate_distances(electrons.position, nuclei.position)
    
        @nnx.vmap(in_axes=(0, None, None), out_axes=(0, 0))
        def calc_slater(slater: SlaterDeterminant, distances: Array, spins: Array) -> Array:
            return slater(distances, spins)

        slater_logs, slater_signs = calc_slater(self.slater_determinant, distances, electrons.spin)
        slaters_sum, sign = slater_log_sum_exp(slater_logs, slater_signs, self.slater_strengths)

        # result = jnp.exp(self.jastrow_factor(distances)) * slaters
        result = self.jastrow_factor(distances) + slaters_sum

        return result.squeeze(), sign  # Ensure we return a scalar

    def eval_wrt_electron_position(self, position: Array, nuclei: Nucleus, position_index: int, all_positions: Array, spins: Array) -> Array:
        """
        Helper function to evaluate the wave function with respect to a specific electron's position.
        This is used in the Hamiltonian to compute the kinetic term.
        """
        modified_positions = all_positions.at[position_index].set(position)
        electrons = Electron(position=modified_positions, spin=spins)
        return self(electrons, nuclei)


def init_electrons(nuclei: Nucleus, num_chains: int, rng: Key) -> Electron:
    """
    Initializes electrons around nuclei with random positions and spins.

    Args:
        nuclei: A list of Nucleus objects.
        num_chains: The number of simulation chains.
        rng: A JAX random key.

    Returns:
        An Electron object with initialized positions and spins.
    """
    # Repeat the nuclei positions according to their charges
    positions = jnp.repeat(nuclei.position, nuclei.charge, axis=0)

    # Split the random key for position and spin generation.
    pos_rng, spin_rng = jax.random.split(rng)

    # Generate random positions around the nuclei for each chain.
    # The positions are created with a shape of (num_chains, num_electrons, 3).
    num_electrons = positions.shape[0]
    positions = jnp.expand_dims(positions, axis=0) + jax.random.normal(pos_rng, shape=(num_chains, num_electrons, 3))

    # Generate random spins for each electron in each chain.
    spins = jax.random.choice(spin_rng, jnp.array([-1, 1]), shape=(num_chains, num_electrons))

    print(f"Positions shape: {positions.shape}, Spins shape: {spins.shape}")
    return Electron(position=positions, spin=spins)


def walk_electrons(sigma: Array) -> Callable:
    sampler = blackjax.mcmc.random_walk.normal(sigma)

    def propose(rng_key: Key, electrons: Electron) -> Electron:
        new_positions = sampler(rng_key, electrons.position)

        # Flip the spin of the electrons with a probability of 0.1
        flip_mask = jax.random.uniform(rng_key, shape=electrons.spin.shape) < 0.1
        new_spins = jnp.where(flip_mask, -electrons.spin, electrons.spin)

        return Electron(position=new_positions, spin=new_spins)

    return propose


@nnx.jit(static_argnames=["num_samples"])
def sample_from_wavefunction(wave_function: WaveFunction, nuclei: Nucleus, initial_positions: Electron, num_samples: int, key: Key) -> ArrayLike:
    @jax.jit
    def call_wavefunction_wrapper(electrons: Electron, nuclei: Nucleus, graphdef: nnx.GraphDef, state: nnx.State) -> Array:
        wave_function = nnx.merge(graphdef, state)
        print(f"Electrons shape: {electrons.position.shape}")
        log_psi, _sign = wave_function(electrons, nuclei)
        print(f"Wave function output shape: {log_psi.shape}")
        # return jnp.log(jnp.conjugate(psi) * psi)
        return 2 * log_psi

    init_key, step_key = jax.random.split(key, 2)

    sigma = 0.1
    graphdef, state = nnx.split(wave_function)
    random_walk = blackjax.additive_step_random_walk(
        partial(call_wavefunction_wrapper, nuclei=nuclei, graphdef=graphdef, state=state),
        walk_electrons(sigma)
    )
    # inv_mass_matrix = jnp.ones_like(initial_position) # Example
    # nuts_kernel = blackjax.nuts(call_wavefunction_wrapper, step_size=1.0, inverse_mass_matrix=inv_mass_matrix)

    # Use the window adaptation utility
    # (last_state, tuned_kernel_params), _ = blackjax.window_adaptation(
    #     nuts_kernel,
    #     call_wavefunction_wrapper,
    #     num_adaptation_steps=500, # Number of steps for warm-up
    #     initial_position=initial_position,
    # )

    num_chains = initial_positions.position.shape[0]
    chain_state = jax.vmap(random_walk.init)(initial_positions)
    # chain_state = random_walk.init(electrons)
    step = jax.jit(random_walk.step)
    # step = random_walk.step

    @nnx.scan(in_axes=(0, nnx.Carry), out_axes=(0, 0, nnx.Carry), length=num_samples)
    def scan_step(step_key, chain_state):
        # new_chain_state, info = step(step_key, chain_state)
        sample_keys = jax.random.split(step_key, num_chains)
        new_chain_state, info = jax.vmap(step, axis_size=num_chains)(sample_keys, chain_state)
        electron_state_sample, log_density = new_chain_state
        return electron_state_sample, log_density, new_chain_state

    step_keys = jax.random.split(step_key, num_samples)
    electron_state_samples, log_densities, chain_state = scan_step(step_keys, chain_state)

    return electron_state_samples, log_densities, chain_state

@nnx.jit
@nnx.value_and_grad(has_aux=True)
def log_prob_fn(wave_function: WaveFunction, sample: Electron, nuclei: Nucleus) -> Array:
    eps = 1e-8
    log_psi, _sign = wave_function(sample, nuclei)
    # log_prob = jnp.log(jnp.conjugate(psi) * psi + eps)
    log_prob = 2 * log_psi
    return log_prob, log_psi

@nnx.jit
def local_energy(wave_function: WaveFunction, sample: Electron, nuclei: Nucleus) -> Array:
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
    local_energy, hamiltonian_details = hamiltonian(wave_function, sample, nuclei)

    (log_prob, log_psi), log_grads = log_prob_fn(wave_function, sample, nuclei)
    # print(f"Log prob: {log_prob}")
    # print(f"Psi: {psi}")
    # print(f"Log grads: {log_grads}")

    return local_energy, log_grads, hamiltonian_details, log_psi


@nnx.jit
def optimize_wave_function(wave_function: nnx.Module, optimizer: nnx.Optimizer, samples: Electron, nuclei: Nucleus) -> Array:
    local_energies, log_grads, hamiltonian_details, log_psi = nnx.vmap(local_energy, in_axes=(None, 0, None))(wave_function, samples, nuclei)
    hamiltonian_details = jax.tree.map(lambda x: jnp.mean(x, axis=0), hamiltonian_details)
    # print(f"Log gradients: {log_grads}")
    
    avg_energy = jnp.mean(local_energies)
    centered_energies = local_energies - avg_energy
    print(f"Energy offsets: {centered_energies}")

    loss_gradient = jax.tree.map(
        lambda log_grads: jnp.expand_dims(centered_energies, axis=tuple(range(1, log_grads.ndim))) * log_grads,
        log_grads,
    )
    # print(f"Loss gradient: {loss_gradient}")

    mean_loss_grads = jax.tree.map(
        lambda loss_gradient: jnp.mean(loss_gradient, axis=0),
        loss_gradient,
    )
    # print(f"Mean loss gradients: {mean_loss_grads}")

    optimizer.update(mean_loss_grads)

    return avg_energy, hamiltonian_details, jnp.mean(log_psi)


@nnx.jit(static_argnames=["num_samples"])
def sample_and_optimize_wave_function(wave_function: WaveFunction, optimizer: nnx.Optimizer, electron_positions: Electron, nuclei: Nucleus, num_samples: int, key: Key):
    key, chain_key = jax.random.split(key, 2)

    # Consider restoring from the previous chain state
    samples, log_densities, chain_state = sample_from_wavefunction(wave_function, nuclei, electron_positions, num_samples, chain_key)

    # Discard the burn-in samples from every chain
    burnin = 100
    skip_factor = 2
    samples = jax.tree.map(lambda x: x[burnin::skip_factor], samples)
    log_densities = jax.tree.map(lambda x: x[burnin::skip_factor], log_densities)

    # Flatten the samples and log_densities for easier handlinga to be free of the chain dimension
    samples, log_densities = jax.tree.map(lambda x: einops.rearrange(x, "n c ... -> (n c) ..."), (samples, log_densities))

    avg_energy, hamiltonian_details, log_psi = optimize_wave_function(wave_function, optimizer, samples, nuclei)

    return key, samples, avg_energy, hamiltonian_details, log_psi


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
    # Initialize electrons around the nuclei
    electrons = init_electrons(nuclei, num_chains=num_chains, rng=key)

    # Start the optimization loop
    for step in range(num_steps):
        print(f"Step {step}/{num_steps}")

        key, samples, avg_energy, hamiltonian_details, log_psi = sample_and_optimize_wave_function(
            wave_function, optimizer, electrons, nuclei, num_samples=num_samples, key=key
        )

        if step % 1000 == 0:
            fig = go.Figure(data=[go.Scatter3d(
                x=samples.position[:, 0],
                y=samples.position[:, 1],
                z=samples.position[:, 2],
                mode='markers',
                marker=dict(size=2),
            )])
            fig.show()

        print(f"Average energy step {step}: {avg_energy}")
        print(f"Hamiltonian details step {step}: {hamiltonian_details}")
        print(f"Log psi step {step}: {log_psi}")

if __name__ == "__main__":
    # Setup nuclei
    nuclei_positions = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=jnp.float32)
    nuclei_charges = jnp.array([3, 1], dtype=jnp.int32)
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
    electronic_optimizer = nnx.Optimizer(
        wave_function,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(1e-3),
        )
    )
    train_key = jax.random.key(0)

    total_steps = 50_000
    num_chains = 150

    train_wavefunction(
        wave_function,
        electronic_optimizer,
        nuclei,
        num_steps=total_steps,
        num_samples=1000,
        num_chains=num_chains,
        key=train_key
    )
