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

def hamiltonian(nuclei: Nucleus):
    nuclea_i, nuclea_j = jnp.triu_indices(nuclei.position.shape[0], k=1)
    nuclei_charge = nuclei.charge[nuclea_i] * nuclei.charge[nuclea_j]
    nuclei_distances = jnp.linalg.norm(nuclei.position[nuclea_i] - nuclei.position[nuclea_j], axis=-1)
    nuclea_interaction = jnp.sum(nuclei_charge / nuclei_distances)

    def _inner(electrons: Electron, wave_function) -> Array:
        electron_i, electron_j = jnp.triu_indices(electrons.position.shape[0], k=1)
        electron_distances = jnp.linalg.norm(electrons.position[electron_i] - electrons.position[electron_j], axis=-1)
        electron_repulsion = 1/jnp.sum(electron_distances)

        electron_nuclei_distances = jnp.linalg.norm(
            electrons.position[:, None] - nuclei.position[None, :], axis=-1
        )
        nucleus_electron_interaction = -jnp.sum(nuclei.charge[None, :] / electron_nuclei_distances)

        laplacian = forward_laplacian(wave_function)(electrons)
        laplacian = -0.5 * jnp.sum(laplacian.laplacian)

        return laplacian + electron_repulsion + nucleus_electron_interaction + nuclea_interaction
    return _inner


nuclei_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=jnp.float32)
nuclei_charges = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
nuclei = Nucleus(position=nuclei_positions, charge=nuclei_charges)

electron_positions = jnp.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], dtype=jnp.float32)
electron_spins = jnp.array([[1.0], [-1.0]], dtype=jnp.float32)
electrons = Electron(position=electron_positions, spin=electron_spins)

mu = jnp.array([-1., 0., 1.])
sigma = jnp.array([0.1, 0.2, 0.3])
dist_distrax = distrax.MultivariateNormalDiag(mu, sigma)

# def wave_function(electrons: Electron) -> Array:
#     # A simple wave function, for example, a Gaussian
#     # return jnp.exp(-jnp.sum(electrons.position**2, axis=-1))
#     print("Computing wave function for electrons:", electrons)
#     prob = dist_distrax.log_prob(electrons.position)
#     # print("Got prob: ", prob)
#     return prob

# The following constructs a wave function using a neural network, following the
# terminology described by https://en.wikipedia.org/wiki/Slater_determinant#Multi-particle_case
class Chi(nnx.Module):
    def __init__(self, hdim: int, rngs: nnx.Rngs):
        self.spin_encoder = nnx.Embed(2, features=hdim, rngs=rngs)
        self.position_encoder = nnx.Linear(3, hdim, rngs=rngs)

        # TODO: Implement proper attention
        self.attention = nnx.Linear(hdim, hdim, rngs=rngs)

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

        wave_function_value = jnp.dot(attention_output, attention_output.T)
        return wave_function_value


class SlaterDeterminant(nnx.Module):
    def __init__(self, num_electrons: int, hidden_dim: int, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=num_electrons)
        @nnx.vmap
        def create_chis(rng: Key) -> Chi:
            return Chi(hidden_dim, rng)
        self.chis = create_chis(rngs)

    def __call__(self, electrons: Electron) -> Array:
        @nnx.vmap
        def calc_chis(chi: Chi) -> Array:
            @nnx.vmap
            def _inner(electrons: Electron) -> Array:
                print("Calculating chi for electron:", electrons)
                return chi(electrons)
            return _inner(electrons)

        print(f"Chis: {self.chis}")
        determinant = jnp.linalg.det(calc_chis(self.chis)(electrons))
        return determinant


class WaveFunction(nnx.Module):
    def __init__(self, num_electrons: int, rngs: nnx.Rngs):
        hidden_dim = 32
        self.slater_determinant = SlaterDeterminant(num_electrons, hidden_dim, rngs)
        self.jastrow_strength = nnx.Param(jnp.array(-0.01))

    def __call__(self, electrons: Electron) -> Array:
        # Exponential fall-off Jastrow factor
        # TODO: Tweak this?
        jastrow = jnp.exp(self.jastrow_strength * jnp.sum(electrons.position**2, axis=-1))
        print("Jastrow factor:", jastrow)
        return jastrow * self.slater_determinant(electrons)

wave_function = WaveFunction(num_electrons=electrons.position.shape[0], rngs=nnx.Rngs(jax.random.key(0)))
h = hamiltonian(nuclei)(electrons, wave_function)
print(h)

hamiltonian_fixed_nuclei = hamiltonian(nuclei)


def walk_electrons(sigma: Array) -> Callable:
    sampler = blackjax.mcmc.random_walk.normal(sigma)

    def propose(rng_key: Key, electrons: Electron) -> Electron:
        new_positions = sampler(rng_key, electrons.position)
        return Electron(position=new_positions, spin=jnp.zeros_like(electrons.spin))

    return propose

sigma = 0.2

@nnx.jit(static_argnames=["num_chains", "num_samples"])
def sample_from_wavefunction(wave_function: WaveFunction, num_chains: int, num_samples: int, key: Key) -> ArrayLike:
    random_walk = blackjax.additive_step_random_walk(wave_function, walk_electrons(sigma))

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

chain_key = jax.random.key(0)
samples, log_densities, chain_state = sample_from_wavefunction(wave_function, 4, 100, chain_key)

print(f"Electron samples: {samples}, Log densities: {log_densities}")

    # hamiltonian_value = hamiltonian_fixed_nuclei(electron_state_sample, wave_function)
    # print(f"Hamiltonian: {hamiltonian_value}, Log Density: {log_density}")
