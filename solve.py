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

@jax.tree_util.register_dataclass
@dataclass
class Electron:
    position: Array
    spin: Array

@jax.tree_util.register_dataclass
@dataclass
class Nucleus:
    position: Array
    charge: Array

@jax.tree_util.register_dataclass
@dataclass
class HamiltonianParts:
    kinetic_energy: Array
    potential_energy: Array

    # Partial contributions to the potential energy
    psi: Array
    electron_repulsion: Array
    nucleus_electron_interaction: Array
    nucleus_nucleus_interaction: Array
    
@nnx.jit
def calculate_kinetic_energy(wave_function: "WaveFunction", electrons: Electron, nuclei: Nucleus) -> Array:
    position_indices = jnp.arange(electrons.position.shape[0])

    @nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
    def calc_terms(wave_function: WaveFunction, position: Array, electron_idx: int) -> Array:
        # Evaluate the wave function with respect to the electron's position
        def calc_wrt_position(position: Array, graphdef: nnx.GraphDef, state: nnx.State) -> Array:
            wave_function = nnx.merge(graphdef, state)
            return wave_function.eval_wrt_electron_position(position, nuclei, electron_idx, electrons.position, electrons.spin)

        graphdef, state = nnx.split(wave_function)
        laplacian = forward_laplacian(partial(calc_wrt_position, graphdef=graphdef, state=state))(position)
        return laplacian.laplacian

    terms = calc_terms(wave_function, electrons.position, position_indices)
    print(f"Laplacian terms shape: {terms.shape}")
    return -0.5 * jnp.sum(terms)

@nnx.jit
def hamiltonian(wave_function: "WaveFunction", electrons: Electron, nuclei: Nucleus):
    """
    For a given state |psi> computes: H|psi> / psi (the local energy).
    """
    nuclea_i, nuclea_j = jnp.triu_indices(nuclei.position.shape[0], k=1)
    nuclei_charge = nuclei.charge[nuclea_i] * nuclei.charge[nuclea_j]
    nuclei_distances = jnp.linalg.norm(nuclei.position[nuclea_i] - nuclei.position[nuclea_j], axis=-1)
    nuclea_interaction = jnp.sum(nuclei_charge / nuclei_distances)

    electron_repulsion = 0.0
    if electrons.position.shape[0] > 1:
        electron_i, electron_j = jnp.triu_indices(electrons.position.shape[0], k=1)
        electron_distances = jnp.linalg.norm(electrons.position[electron_i] - electrons.position[electron_j], axis=-1)
        electron_repulsion = jnp.sum(1.0 / electron_distances)

    electron_nuclei_distances = jnp.linalg.norm(
        electrons.position[:, None] - nuclei.position[None, :], axis=-1
    )
    nucleus_electron_interaction = -jnp.sum(nuclei.charge[None, :] / electron_nuclei_distances)

    psi = wave_function(electrons, nuclei)
    kinetic_energy = calculate_kinetic_energy(wave_function, electrons, nuclei) / psi
    potential_energy = electron_repulsion + nucleus_electron_interaction + nuclea_interaction

    result = kinetic_energy + potential_energy
    return result, HamiltonianParts(
        kinetic_energy=kinetic_energy,
        potential_energy=potential_energy,

        electron_repulsion=electron_repulsion,
        nucleus_electron_interaction=nucleus_electron_interaction,
        nucleus_nucleus_interaction=nuclea_interaction,
        psi=psi,
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
    diff = position_a[:, None] - position_b[None, :]
    square_distances = jnp.sum(diff ** 2, axis=-1)
    distances = jnp.sqrt(square_distances)

    if upper_half:
        # Keep only the upper half of the matrix (i.e., distances between different electrons)
        print("Inside upper half calculation")
        upper_indices = jnp.triu_indices(distances.shape[0], k=1)
        distances = distances[upper_indices]
        square_distances = square_distances[upper_indices]

    print(f"Pairwise distances shape: {distances.shape}")
    distances = jnp.reshape(distances, (-1, ))
    square_distances = jnp.reshape(square_distances, (-1, ))
    return distances, square_distances


@nnx.vmap(in_axes=(0, None), out_axes=0)
def calculate_distance_tokens(position_a: Array, position_b: Array) -> Array:
    square_distances = jnp.sum((position_a - position_b) ** 2, axis=-1)
    distances = jnp.sqrt(square_distances)
    return square_distances, distances


class ElectronicAttention(nnx.Module):
    def __init__(self, hdim: int, num_electrons: int, rngs: nnx.Rngs):
        self.spin_encoder = nnx.Embed(2, features=hdim, rngs=rngs)
        self.position_encoder = nnx.Linear(3, hdim, rngs=rngs)

        nucleus_distance_dim = nuclei.position.shape[0] * num_electrons
        self.nucleus_distance_encoder = nnx.Linear(nucleus_distance_dim, hdim, rngs=rngs)

        # self.electron_distance_encoder = None
        # electron_distance_dim = num_electrons * (num_electrons - 1) // 2
        # if electron_distance_dim > 0:
        #     print(f"Electron distance dimension: {electron_distance_dim}")
        #     self.electron_distance_encoder = nnx.Linear(electron_distance_dim, hdim, rngs=rngs)
        #     self.electron_square_distance_encoder = nnx.Linear(electron_distance_dim, hdim, rngs=rngs)
        self.electron_distance_encoder = nnx.Linear(num_electrons, hdim, rngs=rngs)
        self.electron_square_distance_encoder = nnx.Linear(num_electrons, hdim, rngs=rngs)

        # TODO: Implement proper attention
        self.transformer = TransformerStack(
            num_layers=4,
            num_heads=4,
            hidden_dim=hdim,
            rngs=rngs,
        )
        self.down_project = nnx.Linear(hdim, 1, rngs=rngs)

    def __call__(self, electron: Electron, nuclei: Nucleus, extra_features: Array | None = None) -> Array:
        position_embedding = self.position_encoder(electron.position)
        print(f"Position embedding shape: {position_embedding.shape}")

        # Add in distances between electrons and nuclei
        nucleus_electron_distances = calculate_pairwise_distances(electron.position, nuclei.position)
        nucleus_distance_embedding = self.nucleus_distance_encoder(nucleus_electron_distances)

        electron_distance_embedding = jnp.zeros_like(nucleus_distance_embedding)  # Default to zero if no electron distances
        if self.electron_distance_encoder is not None:
            # Add in distances between electrons
            # electron_electron_distances, electron_electron_square_distances = calculate_pairwise_distances(electron.position, electron.position, upper_half=True)
            # electron_distance_embedding = self.electron_distance_encoder(electron_electron_distances)
            # electron_square_distance_embedding = self.electron_square_distance_encoder(electron_electron_square_distances)

            electron_square_distances, electron_distances = calculate_distance_tokens(electron.position, electron.position)
            print(f"Electron distances shape: {electron_distances.shape}, Electron square distances shape: {electron_square_distances.shape}")
            electron_distance_embedding = self.electron_distance_encoder(electron_distances)
            electron_square_distance_embedding = self.electron_square_distance_encoder(electron_square_distances)
            print(f"Electron distance embedding shape: {electron_distance_embedding.shape}, Electron square distance embedding shape: {electron_square_distance_embedding.shape}")

        combined_embedding = position_embedding
        if extra_features is not None:
            combined_embedding += extra_features

        print(f"Combined embedding shape: {combined_embedding.shape}")
        # attention_output = self.transformer(combined_embedding, combined_embedding + electron_distance_embedding)
        attention_output = self.transformer(combined_embedding)

        return self.down_project(attention_output)


# The following constructs a wave function using a neural network, following the
# terminology described by https://en.wikipedia.org/wiki/Slater_determinant#Multi-particle_case
class Chi(nnx.Module):
    def __init__(self, hdim: int, num_electrons: int, rngs: nnx.Rngs):
        self.spin_encoder = nnx.Embed(2, features=hdim, rngs=rngs)
        self.electronic_attention = ElectronicAttention(hdim, num_electrons, rngs)

    def __call__(self, electron: Electron, nuclei: Nucleus) -> Array:
        spin_classes = jnp.squeeze(jnp.where(electron.spin > 0, 1, 0))
        print("Spin classes:", spin_classes)
        spin_embedding = self.spin_encoder(spin_classes)

        return self.electronic_attention(electron, nuclei, extra_features=spin_embedding)


class SlaterDeterminant(nnx.Module):
    def __init__(self, num_electrons: int, hidden_dim: int, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=num_electrons)
        @nnx.vmap(in_axes=(None, 0))
        def create_chis(num_electrons: int, rng: Key) -> Chi:
            return Chi(hidden_dim, num_electrons, rng)
        self.chis = create_chis(num_electrons, rngs)

    def __call__(self, electrons: Electron, nuclei: Nucleus) -> Array:
        @nnx.vmap(in_axes=(0, None), out_axes=0)
        def calc_chis(chi: Chi, electrons: Electron) -> Array:
            # @nnx.vmap
            # def _inner(electrons: Electron) -> Array:
            #     return chi(electrons)
            # return _inner(electrons)
            return chi(electrons, nuclei)

        chi_matrix = calc_chis(self.chis, electrons)
        chi_matrix = jnp.squeeze(chi_matrix, axis=-1)

        print(f"Chi matrix shape: {chi_matrix.shape}")
        determinant = jnp.linalg.det(chi_matrix)
        return determinant


class JastrowFactor(nnx.Module):
    """
        Map from N * R^3 to R, where N is the number of electrons and R^3 is the position of each electron.
    """
    def __init__(self, hdim: int, num_electrons: int, rngs: nnx.Rngs):
        self.electronic_attention = ElectronicAttention(hdim, num_electrons, rngs)

    def __call__(self, electrons: Electron, nuclei: Nucleus) -> Array:
        attention = self.electronic_attention(electrons, nuclei)
        return jnp.sum(attention)  # Consider max pooling / other pooling methods


class WaveFunction(nnx.Module):
    def __init__(self, num_electrons: int, rngs: nnx.Rngs):
        hidden_dim = 32
        self.jastrow_factor = JastrowFactor(num_electrons=num_electrons, hdim=hidden_dim, rngs=rngs)
        # self.slater_determinants = SlaterDeterminant(num_electrons, hidden_dim, rngs)

        num_slaters = 2
        @nnx.split_rngs(splits=num_slaters)
        @nnx.vmap(in_axes=(None, 0))
        def create_slater_determinants(num_electrons: int, rng: Key) -> SlaterDeterminant:
            return SlaterDeterminant(num_electrons, hidden_dim, rng)
        self.slater_determinant = create_slater_determinants(num_electrons, rngs)

        self.slater_strengths = nnx.Param(jnp.ones((num_slaters,)))

    def __call__(self, electrons: Electron, nuclei: Nucleus) -> Array:
        @nnx.vmap(in_axes=(0, None), out_axes=0)
        def calc_slater(slater: SlaterDeterminant, electrons: Electron) -> Array:
            return slater(electrons, nuclei)
        
        slaters = jnp.sum(self.slater_strengths * calc_slater(self.slater_determinant, electrons), axis=0)
        result = self.jastrow_factor(electrons, nuclei) * slaters

        # result = self.jastrow_factor(electrons, nuclei) * self.slater_determinant(electrons, nuclei)
        # result = self.slater_determinant(electrons, nuclei)
        return result.squeeze()  # Ensure we return a scalar

    def eval_wrt_electron_position(self, position: Array, nuclei: Nucleus, position_index: int, all_positions: Array, spins: Array) -> Array:
        """
        Helper function to evaluate the wave function with respect to a specific electron's position.
        This is used in the Hamiltonian to compute the kinetic term.
        """
        modified_positions = all_positions.at[position_index].set(position)
        electrons = Electron(position=modified_positions, spin=spins)
        print(f"Electrons: {electrons}")
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
        psi = wave_function(electrons, nuclei)
        print(f"Wave function output shape: {psi.shape}")
        return jnp.log(jnp.conjugate(psi) * psi)

    init_key, step_key = jax.random.split(key, 2)

    sigma = 0.25
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
    psi = wave_function(sample, nuclei)
    log_prob = jnp.log(jnp.conjugate(psi) * psi)
    return log_prob, psi

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

    (log_prob, psi), log_grads = log_prob_fn(wave_function, sample, nuclei)
    # print(f"Log prob: {log_prob}")
    # print(f"Psi: {psi}")
    # print(f"Log grads: {log_grads}")

    return local_energy, log_grads, hamiltonian_details


@nnx.jit
def optimize_wave_function(wave_function: nnx.Module, optimizer: nnx.Optimizer, samples: Electron, nuclei: Nucleus) -> Array:
    local_energies, log_grads, hamiltonian_details = nnx.vmap(local_energy, in_axes=(None, 0, None))(wave_function, samples, nuclei)
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

    return avg_energy, hamiltonian_details


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

    avg_energy, hamiltonian_details = optimize_wave_function(wave_function, optimizer, samples, nuclei)

    return key, samples, avg_energy, hamiltonian_details


# Setup nuclei and electrons
# nuclei_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)
# nuclei_charges = jnp.array([1.0, 1.0], dtype=jnp.float32)
nuclei_positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
nuclei_charges = jnp.array([2], dtype=jnp.int32)
nuclei = Nucleus(position=nuclei_positions, charge=nuclei_charges)

electron_positions = jnp.array([[0.5, -0.4, 0.2], [1.0, 0.2, 0.6]], dtype=jnp.float32)
electron_spins = jnp.array([[-1], [1]], dtype=jnp.int32)
electrons = Electron(position=electron_positions, spin=electron_spins)

# electron_positions = jnp.array([[0.5, -0.4, 0.2]], dtype=jnp.float32)
# electron_spins = jnp.array([[-1]], dtype=jnp.int32)
# electrons = Electron(position=electron_positions, spin=electron_spins)


# Initialize the wave function
wave_function = WaveFunction(num_electrons=electrons.position.shape[0], rngs=nnx.Rngs(jax.random.key(0)))
direct_wave_func_eval = wave_function(electrons, nuclei)
print(f"Direct wave function evaluation: {direct_wave_func_eval}")

h, h_details = hamiltonian(wave_function, electrons, nuclei)
print(f"Hamiltonian: {h}, details: {h_details}")

# Start the wave function optimization
optimizer = nnx.Optimizer(wave_function, optax.adamw(1e-3))
train_key = jax.random.key(0)

total_steps = 50_000

num_chains = 150
electrons = init_electrons(nuclei, num_chains=num_chains, rng=train_key)  # Shape (num_chains, num_electrons, 3)
for step in range(total_steps):
    print(f"Step {step}/{total_steps}")

    train_key, samples, avg_energy, hamiltonian_details = sample_and_optimize_wave_function(
        wave_function, optimizer, electrons, nuclei, num_samples=2000, key=train_key
    )

    if step % 1000 == 0:
        fig = go.Figure(data=[go.Scatter3d(
            x=samples.position[:, 0, 0],
            y=samples.position[:, 0, 1],
            z=samples.position[:, 0, 2],
            mode='markers',
            marker=dict(size=2),
        )])
        fig.show()

    print(f"Average energy step {step}: {avg_energy}")
    print(f"Hamiltonian details step {step}: {hamiltonian_details}")
