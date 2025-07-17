import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from jaxtyping import Array, Key, ArrayLike
from typing import List, Optional, Callable
from folx import forward_laplacian
import blackjax
import distrax
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

def hamiltonian(nuclei: Nucleus, wave_function: nnx.Module):
    nuclea_i, nuclea_j = jnp.triu_indices(nuclei.position.shape[0], k=1)
    nuclei_charge = nuclei.charge[nuclea_i] * nuclei.charge[nuclea_j]
    nuclei_distances = jnp.linalg.norm(nuclei.position[nuclea_i] - nuclei.position[nuclea_j], axis=-1)
    nuclea_interaction = jnp.sum(nuclei_charge / nuclei_distances)

    @jax.jit
    def call_wavefunction_wrapper(graphdef: nnx.GraphDef, state: nnx.State, electrons: Electron) -> Array:
        model = nnx.merge(graphdef, state)
        return model(electrons)
    graphdef, state = nnx.split(wave_function)

    @jax.jit
    def _inner(electrons: Electron) -> Array:
        electron_repulsion = 0.0
        if electrons.position.shape[0] > 1:
            electron_i, electron_j = jnp.triu_indices(electrons.position.shape[0], k=1)
            electron_distances = jnp.linalg.norm(electrons.position[electron_i] - electrons.position[electron_j], axis=-1)
            electron_repulsion = 1/jnp.sum(electron_distances)

        electron_nuclei_distances = jnp.linalg.norm(
            electrons.position[:, None] - nuclei.position[None, :], axis=-1
        )
        nucleus_electron_interaction = -jnp.sum(nuclei.charge[None, :] / electron_nuclei_distances)

        laplacian = forward_laplacian(call_wavefunction_wrapper)(graphdef, state, electrons)
        laplacian = -0.5 * jnp.sum(laplacian.laplacian)

        # jax.debug.print("Laplacian: {laplacian}, Electron repulsion: {electron_repulsion}, nucleus-electron interaction: {nucleus_electron_interaction}, nucleus-nucleus interaction: {nuclea_interaction}",
        #                 laplacian=laplacian, electron_repulsion=electron_repulsion,
        #                 nucleus_electron_interaction=nucleus_electron_interaction,
        #                 nuclea_interaction=nuclea_interaction)
        return laplacian + electron_repulsion + nucleus_electron_interaction + nuclea_interaction
    return _inner


# nuclei_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)
# nuclei_charges = jnp.array([1.0, 1.0], dtype=jnp.float32)
nuclei_positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
nuclei_charges = jnp.array([1.0], dtype=jnp.float32)
nuclei = Nucleus(position=nuclei_positions, charge=nuclei_charges)

electron_positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
electron_spins = jnp.array([[-1.0]], dtype=jnp.float32)
electrons = Electron(position=electron_positions, spin=electron_spins)


# The following constructs a wave function using a neural network, following the
# terminology described by https://en.wikipedia.org/wiki/Slater_determinant#Multi-particle_case
class Chi(nnx.Module):
    def __init__(self, hdim: int, rngs: nnx.Rngs):
        self.spin_encoder = nnx.Embed(2, features=hdim, rngs=rngs)
        self.position_encoder = nnx.Linear(3, hdim, rngs=rngs)

        # TODO: Implement proper attention
        self.attention = nnx.Linear(hdim, hdim, rngs=rngs)
        self.down_project = nnx.Linear(hdim, 1, rngs=rngs)

    def __call__(self, electron: Electron) -> Array:
        spin_classes = jnp.where(electron.spin > 0, 1, 0)
        print("Spin classes:", spin_classes)
        spin_embedding = self.spin_encoder(spin_classes)
        spin_embedding = jnp.squeeze(spin_embedding, axis=-2)
        position_embedding = self.position_encoder(electron.position)
        print(f"Position embedding shape: {position_embedding.shape}, Spin embedding shape: {spin_embedding.shape}")

        combined_embedding = spin_embedding + position_embedding
        print(f"Combined embedding shape: {combined_embedding.shape}")
        attention_output = self.attention(combined_embedding)

        return self.down_project(attention_output)


class SlaterDeterminant(nnx.Module):
    def __init__(self, num_electrons: int, hidden_dim: int, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=num_electrons)
        @nnx.vmap
        def create_chis(rng: Key) -> Chi:
            return Chi(hidden_dim, rng)
        self.chis = create_chis(rngs)

    def __call__(self, electrons: Electron) -> Array:
        @nnx.vmap(in_axes=(0, None), out_axes=0)
        def calc_chis(chi: Chi, electrons: Electron) -> Array:
            @nnx.vmap
            def _inner(electrons: Electron) -> Array:
                return chi(electrons)
            return _inner(electrons)

        chi_matrix = calc_chis(self.chis, electrons)
        # chi_matrix = jnp.squeeze(chi_matrix, axis=-1)

        print(f"Chi matrix shape: {chi_matrix.shape}")
        determinant = jnp.linalg.det(chi_matrix)
        return determinant


class JastrowFactor(nnx.Module):
    """
        Map from N * R^3 to R, where N is the number of electrons and R^3 is the position of each electron.

        For now just make sure the wave-function falls off at infinity.
    """
    def __init__(self):
        # TODO: Make this a more complex function that has more parameters
        self.jastrow_strength = nnx.Param(jnp.array(-2.0))

    def __call__(self, electrons: Electron) -> Array:
        max_pos = jnp.max(jnp.abs(electrons.position))
        return jnp.exp(self.jastrow_strength * max_pos)


class WaveFunction(nnx.Module):
    def __init__(self, num_electrons: int, rngs: nnx.Rngs):
        hidden_dim = 32
        self.jastrow_factor = JastrowFactor()
        self.slater_determinant = SlaterDeterminant(num_electrons, hidden_dim, rngs)

    def __call__(self, electrons: Electron) -> Array:
        result = self.jastrow_factor(electrons) * self.slater_determinant(electrons)
        return result.squeeze()  # Ensure we return a scalar


def walk_electrons(sigma: Array) -> Callable:
    sampler = blackjax.mcmc.random_walk.normal(sigma)

    def propose(rng_key: Key, electrons: Electron) -> Electron:
        new_positions = sampler(rng_key, electrons.position)
        return Electron(position=new_positions, spin=jnp.zeros_like(electrons.spin))

    return propose

sigma = 1.0

@nnx.jit(static_argnames=["num_chains", "num_samples"])
def sample_from_wavefunction(wave_function: WaveFunction, num_chains: int, num_samples: int, key: Key) -> ArrayLike:
    @jax.jit
    def call_wavefunction_wrapper(electrons: Electron, graphdef: nnx.GraphDef, state: nnx.State) -> Array:
        wave_function = nnx.merge(graphdef, state)
        print(f"Electrons shape: {electrons.position.shape}")
        psi = wave_function(electrons)
        print(f"Wave function output shape: {psi.shape}")
        return jnp.log(jnp.conjugate(psi) * psi)

    graphdef, state = nnx.split(wave_function)
    random_walk = blackjax.additive_step_random_walk(
        partial(call_wavefunction_wrapper, graphdef=graphdef, state=state),
        walk_electrons(sigma)
    )
    # inv_mass_matrix = jnp.ones_like(initial_position) # Example
    # nuts_kernel = blackjax.nuts(logprob_fn, step_size=1.0, inverse_mass_matrix=inv_mass_matrix)

    # # Use the window adaptation utility
    # (last_state, tuned_kernel_params), _ = blackjax.window_adaptation(
    #     blackjax.nuts,
    #     logprob_fn,
    #     num_adaptation_steps=500, # Number of steps for warm-up
    #     initial_position=initial_position,
    # )

    chain_state = jax.vmap(random_walk.init, axis_size=num_chains, in_axes=None)(electrons)
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

    step_keys = jax.random.split(key, num_samples)
    electron_state_samples, log_densities, chain_state = scan_step(step_keys, chain_state)

    return electron_state_samples, log_densities, chain_state

@nnx.jit
@nnx.value_and_grad(has_aux=True)
def log_prob_fn(wave_function: WaveFunction, sample: Electron) -> Array:
    psi = wave_function(sample)
    log_prob = jnp.log(jnp.conjugate(psi) * psi)
    return log_prob, psi

@nnx.jit
def local_energy(wave_function: WaveFunction, sample: Electron) -> Array:
    print(f"Sample: {sample}")
    hamiltonian_value = hamiltonian_fixed_nuclei_and_wave(sample)

    (log_prob, psi), log_grads = log_prob_fn(wave_function, sample)
    print(f"Log prob: {log_prob}")
    print(f"Psi: {psi}")
    print(f"Log grads: {log_grads}")


    # Follows because we sample the samples according to the wave function
    # i.e. the integral we evaluate is already weighed by a |psi|^2
    # So we write:
    #   <L> = \int dx psi*(x) (H psi)(x) / \int dx psi*(x)psi(x)
    # WRITE:
    #   psi*(x) (H psi(x)) = (psi*(x) * psi(x)) * (H psi)(x)) / psi(x) (multiply and divide by psi(x))
    #                      = |psi(x)|^2 * Elocal(x)
    local_energy = hamiltonian_value / psi

    # Notice that we do not need to normalize the wave function here, because
    # we are already sampling according to the wave function
    # Exactly because the norm factor is |psi|^2
    # norm_factor = jnp.conjugate(psi) * psi

    return local_energy, log_grads


@nnx.jit
def optimize_wave_function(wave_function: nnx.Module, optimizer: nnx.Optimizer, samples: Electron):
    local_energies, log_grads = nnx.vmap(local_energy, in_axes=(None, 0))(wave_function, samples)
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

    return avg_energy


wave_function = WaveFunction(num_electrons=electrons.position.shape[0], rngs=nnx.Rngs(jax.random.key(0)))
direct_wave_func_eval = wave_function(electrons)
print(f"Direct wave function evaluation: {direct_wave_func_eval}")

h = hamiltonian(nuclei, wave_function)(electrons)
print(f"Hamiltonian: {h}")

optimizer = nnx.Optimizer(wave_function, optax.adamw(1e-3))
train_key = jax.random.key(0)

total_steps = 5
for step in range(total_steps):
    print(f"Step {step}/{total_steps}")
    train_key, chain_key = jax.random.split(train_key, 2)

    hamiltonian_fixed_nuclei_and_wave = hamiltonian(nuclei, wave_function)

    # Consider restoring from the previous chain state
    samples, log_densities, chain_state = sample_from_wavefunction(wave_function, 10, 10_000, chain_key)

    # Discard the burn-in samples from every chain
    burnin = 100
    skip_factor = 2
    samples = jax.tree.map(lambda x: x[burnin::skip_factor], samples)
    log_densities = jax.tree.map(lambda x: x[burnin::skip_factor], log_densities)

    # Flatten the samples and log_densities for easier handlinga to be free of the chain dimension
    samples, log_densities = jax.tree.map(lambda x: einops.rearrange(x, "n c ... -> (n c) ..."), (samples, log_densities))

    fig = go.Figure(data=[go.Scatter3d(
        x=samples.position[:, 0, 0],
        y=samples.position[:, 0, 1],
        z=samples.position[:, 0, 2],
        mode='markers',
        marker=dict(size=2),
    )])
    fig.show()

    avg_energy = optimize_wave_function(wave_function, optimizer, samples)
    print(f"Average energy step {step}: {avg_energy}")
