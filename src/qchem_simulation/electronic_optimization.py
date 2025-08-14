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
from .utils import Electron, Nucleus
import chex


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
class Distance:
    magnitude: Array  # Shape (n, n)
    direction: Array  # Unit vector shape (n, n, 3)

@jax.tree_util.register_dataclass
@dataclass
class Distances:
    electron_distances: Distance
    nuclei_distances: Distance
    electron_nuclei_distances: Distance


# @nnx.jit
def calculate_kinetic_energy(wave_function: "WaveFunction", electrons: Electron, nuclei: Nucleus) -> Array:
    position_indices = jnp.arange(electrons.position.shape[0])

    @nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
    def calc_terms(wave_function: WaveFunction, position: Array, electron_idx: int) -> Array:
        # Evaluate the wave function with respect to the electron's position
        def calc_wrt_position(position: Array, graphdef: nnx.GraphDef, state: nnx.State) -> Array:
            wave_function = nnx.merge(graphdef, state)
            amplitude, _sign, _ortho_measure = wave_function.eval_wrt_electron_position(position, nuclei, electron_idx, electrons.position, electrons.spin)
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

def calculate_kinetic_energy_hutch(wave_function: "WaveFunction", electrons: Electron, nuclei: Nucleus, key: Key, k: int = 10) -> Array:
    position_indices = jnp.arange(electrons.position.shape[0])

    @nnx.vmap(in_axes=(None, 0, 0, 0), out_axes=0)
    def calc_terms(wave_function: WaveFunction, position: Array, electron_idx: int, key: Key) -> Array:
        # Evaluate the wave function with respect to the electron's position
        def calc_wrt_position(position: Array, graphdef: nnx.GraphDef, state: nnx.State) -> Array:
            wave_function = nnx.merge(graphdef, state)
            amplitude, _sign, _ortho_measure = wave_function.eval_wrt_electron_position(position, nuclei, electron_idx, electrons.position, electrons.spin)
            return amplitude

        graphdef, state = nnx.split(wave_function)
        grad_func = jax.grad(partial(calc_wrt_position, graphdef=graphdef, state=state))
        grads = grad_func(position)

        def _iteration(key: Key) -> Array:
            # 2. Generate a random vector `v`
            v = jax.random.normal(key, shape=position.shape)
            # v = jax.random.rademacher(key, shape=position.shape).astype(jnp.float32)

            _primals_out, hvp = jax.jvp(grad_func, (position,), (v,))
            laplacian_estimate = jnp.dot(v, hvp)
            return laplacian_estimate

        iteration_keys = jax.random.split(key, k)
        laplacian_estimate = jnp.mean(jax.vmap(_iteration, axis_size=k)(iteration_keys))

        # Use the identity: ∇^2 psi = psi * (∇^2 log(psi) + |∇log(psi)|^2)
        # We actually want to calculate (∇^2 psi)/psi (the local energy, and from linearity of the Laplacian term
        # in the kinetic energy). Therefore, we calculate: (∇^2 psi)/psi = (∇^2 log(psi) + |∇log(psi)|^2)
        # where |∇log(psi)|^2 = J \dot J^T where J is the Jacobian of the wave function
        # Notice that since the laplacian is linear, the sign of psi will cancel out in (∇^2 psi)/psi.
        amplitude = laplacian_estimate + jnp.dot(grads, grads)
        return amplitude

    keys = jax.random.split(key, electrons.position.shape[0])
    terms = calc_terms(wave_function, electrons.position, position_indices, keys)
    # print(f"Laplacian terms shape: {terms.shape}")
    return -0.5 * jnp.sum(terms)

def calculate_kinetic_energy_hutchpp(wave_function: "WaveFunction", electrons: Electron, nuclei: Nucleus, key: Key, k: int = 9) -> Array:
    position_indices = jnp.arange(electrons.position.shape[0])

    @nnx.vmap(in_axes=(None, 0, 0, 0), out_axes=0)
    def calc_terms(wave_function: WaveFunction, position: Array, electron_idx: int, key: Key) -> Array:
        # Evaluate the wave function with respect to the electron's position
        def calc_wrt_position(position: Array, graphdef: nnx.GraphDef, state: nnx.State) -> Array:
            wave_function = nnx.merge(graphdef, state)
            amplitude, _sign, _ortho_measure = wave_function.eval_wrt_electron_position(position, nuclei, electron_idx, electrons.position, electrons.spin)
            return amplitude

        graphdef, state = nnx.split(wave_function)
        grad_func = jax.grad(partial(calc_wrt_position, graphdef=graphdef, state=state))
        grads = grad_func(position)

        @jax.vmap
        def operator(v: Array) -> Array:
            _primals_out, hvp = jax.jvp(grad_func, (position,), (v,))
            return hvp
        
        m = k // 3
        samples = jax.random.normal(key, shape=(position.shape[0], 2 * m))
        s = samples[:, :m]
        g = samples[:, m:]

        q, _ = jnp.linalg.qr(operator(s))
        # Tr(Q^T A Q)
        qr_part = jnp.einsum("ij,ji", q.T, operator(q))  # Computes the trace

        r = jnp.eye(q.shape[0]) - q @ q.T
        # Tr(G^T R A R^T G) / k
        hutch_correction = jnp.einsum("ij,ji", g.T @ r, operator(r @ g)) / k

        laplacian_estimate = qr_part + hutch_correction

        # Use the identity: ∇^2 psi = psi * (∇^2 log(psi) + |∇log(psi)|^2)
        # We actually want to calculate (∇^2 psi)/psi (the local energy, and from linearity of the Laplacian term
        # in the kinetic energy). Therefore, we calculate: (∇^2 psi)/psi = (∇^2 log(psi) + |∇log(psi)|^2)
        # where |∇log(psi)|^2 = J \dot J^T where J is the Jacobian of the wave function
        # Notice that since the laplacian is linear, the sign of psi will cancel out in (∇^2 psi)/psi.
        amplitude = laplacian_estimate + jnp.dot(grads, grads)
        return amplitude

    keys = jax.random.split(key, electrons.position.shape[0])
    terms = calc_terms(wave_function, electrons.position, position_indices, keys)
    # print(f"Laplacian terms shape: {terms.shape}")
    return -0.5 * jnp.sum(terms)


# @nnx.jit
def hamiltonian(wave_function: "WaveFunction", electrons: Electron, nuclei: Nucleus, key: Key):
    """
    For a given state |psi> computes: H|psi> / psi (the local energy).
    """
    eps = 1e-12

    nuclea_i, nuclea_j = jnp.triu_indices(nuclei.position.shape[0], k=1)
    nuclei_charge = nuclei.charge[nuclea_i] * nuclei.charge[nuclea_j]
    nuclei_distances = jnp.linalg.norm(nuclei.position[nuclea_i] - nuclei.position[nuclea_j], axis=-1)
    nuclea_interaction = jnp.sum(nuclei_charge / (nuclei_distances + eps))

    electron_repulsion = 0.0
    if electrons.position.shape[0] > 1:
        electron_i, electron_j = jnp.triu_indices(electrons.position.shape[0], k=1)
        electron_distances = jnp.linalg.norm(electrons.position[electron_i] - electrons.position[electron_j], axis=-1)
        electron_repulsion = jnp.sum(1.0 / (electron_distances + eps))

    electron_nuclei_distances = jnp.linalg.norm(
        electrons.position[:, None] - nuclei.position[None, :], axis=-1
    )
    nucleus_electron_interaction = -jnp.sum(nuclei.charge[None, :] / (electron_nuclei_distances + eps))

    # The wave_function returns the log-probability, and the calculated kinetic energy is
    # already normalized by psi. See the comments in the `calculate_kinetic_energy` function.
    # kinetic_energy = calculate_kinetic_energy(wave_function, electrons, nuclei)
    # kinetic_energy = calculate_kinetic_energy_hutch(wave_function, electrons, nuclei, key)
    kinetic_energy = calculate_kinetic_energy_hutchpp(wave_function, electrons, nuclei, key)
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
        # h = key_values
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
            # h = layer(h, query)
            h = layer(h)
        return h



def calculate_pairwise_distances(position_a: Array, position_b: Array, upper_half: bool = False) -> Distance:
    """
    Calculate distance features between two sets of positions.
    Returns the squared distances and the inverse distances.

    Rows will be from position_a, and columns will be from position_b.
    """
    eps = 1e-12
    diff = position_a[:, None] - position_b[None, :]
    square_distances = jnp.sum(jnp.square(diff), axis=-1)
    distances = jnp.sqrt(square_distances + eps)

    unit_vectors = diff / distances[:, :, None]

    if upper_half:
        # Keep only the upper half of the matrix (i.e., distances between different electrons)
        # print("Inside upper half calculation")
        upper_indices = jnp.triu_indices(distances.shape[0], k=1)
        distances = jnp.reshape(distances[upper_indices], (-1, ))
        square_distances = jnp.reshape(square_distances[upper_indices], (-1, ))

    return Distance(
        magnitude=distances,
        direction=unit_vectors,
    )


def calculate_distances(electron_positions: Array, nucleus_positions: Array) -> Distances:
    """
    Calculate distances between electrons and nuclei.
    Returns a Distances object containing:
        - electron_distances: Distances between electrons
        - nuclei_distances: Distances between nuclei
        - electron_nuclei_distances: Distances between electrons and nuclei
    """
    electron_distances = calculate_pairwise_distances(electron_positions, electron_positions)
    nucleus_distances = calculate_pairwise_distances(nucleus_positions, nucleus_positions, upper_half=True)
    # Notice that the order here is important, electrons should be rows, nuclei columns
    # This is to make the electronic attention mechanism work correctly with electron permutations
    electron_nuclei_distances = calculate_pairwise_distances(electron_positions, nucleus_positions)

    return Distances(
        electron_distances=electron_distances,
        nuclei_distances=nucleus_distances,
        electron_nuclei_distances=electron_nuclei_distances,
    )


class DistanceEncoder(nnx.Module):
    def __init__(self, hidden_dim: int, buckets: int, rngs: nnx.Rngs, cutoff: float = 10.0):
        self.centers = jnp.linspace(0.0, cutoff, buckets)
        self.sigma = (self.centers[1] -  self.centers[0]) * 1.5

        self.rbf_proj = nnx.Linear(buckets, buckets, rngs=rngs)
        self.direction_encoder = nnx.Linear(3, buckets, rngs=rngs, use_bias=False)
        self.final_proj = nnx.Linear(2 * buckets, hidden_dim, rngs=rngs)

    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def _rbf_encoding(self, magnitudes: Array):
        """
        Radial basis function (RBF) encoding for distance magnitudes.
        """
        chex.assert_shape(magnitudes, (None,))  # Expecting a 1D array of magnitudes
        return jnp.exp(-0.5 * ((magnitudes[:, None] - self.centers[None, :]) / self.sigma) ** 2)

    def __call__(self, distance: Distance):
        chex.assert_shape(distance.magnitude, (None, None))
        chex.assert_shape(distance.direction, (None, None, 3))

        rbf_proj = self.rbf_proj(self._rbf_encoding(distance.magnitude))
        direction_encoding = self.direction_encoder(distance.direction)
        combined_encoding = jnp.concat([rbf_proj, direction_encoding], axis=-1)

        distance_mean = jnp.mean(combined_encoding, axis=1)  # Mean to make sure the distance encoding will be permutation equivalent
        distance_encoding = self.final_proj(distance_mean)

        chex.assert_shape(distance_encoding, (None, self.final_proj.out_features))

        return distance_encoding


@nnx.dataclass
class DistanceEncoders(nnx.Object):
    electron: DistanceEncoder
    electron_nuclei: DistanceEncoder
    # nuclei: DistanceEncoder


class ElectronicAttention(nnx.Module):
    def __init__(self, hdim: int, num_electrons: int, num_nuclei: int, rngs: nnx.Rngs):
        # Set up the nuclei-nuclei distance encoder
        self.nuclei_distance_encoder = None
        nuclei_pairs = num_nuclei * (num_nuclei - 1) // 2
        if nuclei_pairs > 0:
            self.nuclei_distance_encoder = nnx.Linear(nuclei_pairs, hdim, rngs=rngs)

        self.transformer = TransformerStack(
            num_layers=4,
            num_heads=2,
            hidden_dim=hdim,
            rngs=rngs,
        )
        # self.down_project = nnx.Linear(hdim, num_electrons, rngs=rngs)

    def __call__(self, distances: Distances, distance_encoders: DistanceEncoders, extra_features: Array | None = None) -> Array:
        # print("Electron distances:", distances.electron_distances.shape)
        # print("Nuclei distances:", distances.nuclei_distances.shape)
        # print("Electron-nuclei distances:", distances.electron_nuclei_distances.shape)

        # In order to make the attention mechanism permutation-equivariant, we need to sort the distances
        # distances = jax.tree.map(jnp.sort, distances)

        electron_distance_embedding = distance_encoders.electron(distances.electron_distances)
        electron_nuclei_distance_embedding = distance_encoders.electron_nuclei(distances.electron_nuclei_distances)

        distance_embedding = electron_distance_embedding + electron_nuclei_distance_embedding
        print(f"Distance embedding shape: {distance_embedding.shape}")

        if self.nuclei_distance_encoder is not None:
            # TODO(knielsen): Handle this with a proper distance encoder
            nucleus_distance_embedding = self.nuclei_distance_encoder(distances.nuclei_distances.magnitude)
            distance_embedding = distance_embedding + nucleus_distance_embedding[None, :]

        combined_embedding = distance_embedding
        if extra_features is not None:
            # print(f"Extra features shape: {extra_features.shape}")
            chex.assert_equal_shape([combined_embedding, extra_features])
            combined_embedding += extra_features

        print(f"Combined embedding shape: {combined_embedding.shape}")

        attention_output = self.transformer(combined_embedding)
        # attention_output = self.transformer(combined_embedding, electron_distance_embedding)

        return attention_output

        # # Produce a (num_electrons, num_electrons) array
        # # It will have the feature that swapping an electron position i with j, will switch the
        # # output rows.
        # return self.down_project(attention_output)


class OrbitalHead(nnx.Module):
    def __init__(self, hdim: int, orbital_dim: int, rngs: nnx.Rngs):
        self.dense = nnx.Linear(hdim, orbital_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.dense(x)


# The following constructs a wave function using a neural network, following the
# terminology described by https://en.wikipedia.org/wiki/Slater_determinant#Multi-particle_case
# Split the Slater determinant into one for each spin, as interchanging electrons of opposite spin
# will not cause any anti-symmetry
class OrbitalNetwork(nnx.Module):
    def __init__(self, hdim: int, num_electrons: int, num_nuclei: int, rngs: nnx.Rngs):
        self.spin_encoder = nnx.Embed(2, features=hdim, rngs=rngs)
        self.electronic_attention = ElectronicAttention(hdim, num_electrons, num_nuclei, rngs=rngs)

        self.up_head = OrbitalHead(hdim, num_electrons, rngs=rngs)
        self.down_head = OrbitalHead(hdim, num_electrons, rngs=rngs)

    def __call__(self, distances: Distances, spins: Array, distance_encoders: DistanceEncoders) -> Array:
        chex.assert_shape(spins, (None, 1))  # Expecting spins to be a column vector

        spin_classes = jnp.where(spins > 0, 1, 0)
        # print("Spin classes:", spin_classes)
        spin_embedding = jnp.squeeze(self.spin_encoder(spin_classes), axis=-2)

        attention = self.electronic_attention(distances, distance_encoders=distance_encoders, extra_features=spin_embedding)
        print(f"Attention shape: {attention.shape}")

        # Ensure we create proper nodes in the orbital matrix for different spins
        orbital_matrix = jnp.where(spins > 0, self.up_head(attention), self.down_head(attention))

        orthonormality_measure = jnp.sum((orbital_matrix.T @ orbital_matrix - jnp.eye(orbital_matrix.shape[0])) ** 2, axis=(-1, -2))

        return orbital_matrix, orthonormality_measure


class SlaterDeterminant(nnx.Module):
    def __init__(self, num_electrons: int, num_nuclei: int, hidden_dim: int, rngs: nnx.Rngs):
        self.orbital_network = OrbitalNetwork(hidden_dim, num_electrons, num_nuclei, rngs=rngs)

    def __call__(self, distances: Array, spins: Array, distance_encoders: DistanceEncoders) -> Array:
        orbitals, orthonormality_measure = self.orbital_network(distances, spins, distance_encoders=distance_encoders)
        print(f"Orbital matrix shape: {orbitals.shape}")
        log_det = jnp.linalg.slogdet(orbitals, method="qr")  # log determinant
        print(f"Log-Determinant value: {log_det}")
        return log_det.logabsdet, log_det.sign, orthonormality_measure


class CuspElectrons(nnx.Module):
    """
    Models the Kato cusp condition (https://en.wikipedia.org/wiki/Kato_theorem) for electrons.
    """
    def __init__(self, num_electrons: int):
        electron_pairs = (num_electrons * (num_electrons - 1)) // 2
        self.decay_strengths = None
        if electron_pairs > 0:
            self.decay_strengths = nnx.Param(jnp.ones((electron_pairs,)))

    def __call__(self, electron_distances: Array, spins: Array) -> Array:
        chex.assert_rank(spins, 2)
        chex.assert_axis_dimension(spins, 1, 1)
        chex.assert_axis_dimension(spins, 0, electron_distances.shape[0])
        chex.assert_shape(electron_distances, (spins.shape[0], spins.shape[0]))

        if self.decay_strengths is None:
            return 0.0

        upper_indices = jnp.triu_indices(electron_distances.shape[0], k=1)

        flat_pairwise_distances = jnp.reshape(electron_distances[upper_indices], (-1, ))

        spins = jnp.squeeze(spins, axis=-1)
        spin_correlations = spins[:, None] == spins[None, :]
        cusp_values = jnp.where(spin_correlations, 0.25, 0.5)
        flat_cusp_values = jnp.reshape(cusp_values[upper_indices], (-1, ))

        # log-Jastrow factor on the form: \sum_{i, j < i} (1/2) * d_{ij} / (1 + decay_strength_{ij} * d_{ij})
        # where b is the decay strength
        jastrow_factor = jnp.sum(
            (flat_cusp_values * flat_pairwise_distances) / (1.0 + self.decay_strengths * flat_pairwise_distances),
            axis=(-1,),
        )
        return jastrow_factor


class CuspElectronNuclei(nnx.Module):
    """
    Models the Kato cusp condition (https://en.wikipedia.org/wiki/Kato_theorem) for electrons.
    """
    def __init__(self, num_electrons: int, num_nuclei: int):
        self.num_electrons = num_electrons
        self.num_nuclei = num_nuclei

        self.decay_strengths = nnx.Param(jnp.ones((num_electrons, num_nuclei)))

    def __call__(self, electron_nuclei_distances: Array, nuclei_charges: Array):
        chex.assert_shape(electron_nuclei_distances, (self.num_electrons, self.num_nuclei))
        chex.assert_shape(nuclei_charges, (self.num_nuclei,))

        # log-Jastrow factor on the form: \sum_{i, j} -(nuclei_charges{i}) * d_{ij} / (1 + decay_strength_{ij} * d_{ij})
        # where b is the decay strength
        jastrow_factor = -jnp.sum(
            (nuclei_charges[None, :] * electron_nuclei_distances) / (1.0 + self.decay_strengths * electron_nuclei_distances),
            axis=(-1, -2),
        )
        return jastrow_factor


class JastrowFactor(nnx.Module):
    """
        Map from N * R^3 to R, where N is the number of electrons and R^3 is the position of each electron.
    """
    def __init__(self, hdim: int, num_electrons: int, num_nuclei: int, rngs: nnx.Rngs):
        self.electrons_cusp = CuspElectrons(num_electrons)
        self.nuclei_cusp = CuspElectronNuclei(num_electrons, num_nuclei)

        self.electronic_attention = ElectronicAttention(hdim, num_electrons, num_nuclei, rngs=rngs)

    def __call__(self, distances: Distances, nuclei_charges: Array, electron_spins: Array, distance_encoders: DistanceEncoders) -> Array:
        electron_cusp = self.electrons_cusp(distances.electron_distances.magnitude, electron_spins)
        nuclei_cusp = self.nuclei_cusp(distances.electron_nuclei_distances.magnitude, nuclei_charges)

        corrections = jnp.mean(self.electronic_attention(distances, distance_encoders=distance_encoders))
        return electron_cusp + nuclei_cusp + corrections


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
        self.num_electrons = num_electrons
        self.num_nuclei = num_nuclei

        hidden_dim = 64
        self.distance_encoders = DistanceEncoders(
            electron=DistanceEncoder(hidden_dim, buckets=64, rngs=rngs),
            electron_nuclei=DistanceEncoder(hidden_dim, buckets=64, rngs=rngs)
        )

        self.jastrow_factor = JastrowFactor(num_electrons=num_electrons, num_nuclei=num_nuclei, hdim=hidden_dim, rngs=rngs)

        num_slaters = 1
        @nnx.split_rngs(splits=num_slaters)
        @nnx.vmap(in_axes=(None, None, 0))
        def create_slater_determinants(num_electrons: int, num_nuclei: int, rng: Key) -> SlaterDeterminant:
            return SlaterDeterminant(num_electrons, num_nuclei, hidden_dim, rngs=rng)
        self.slater_determinant = create_slater_determinants(num_electrons, num_nuclei, rngs)

        self.slater_strengths = nnx.Param(jnp.ones((num_slaters,), dtype=jnp.float32))

    def __call__(self, electrons: Electron, nuclei: Nucleus) -> Array:
        chex.assert_shape(electrons.position, (self.num_electrons, 3))
        chex.assert_shape(electrons.spin, (self.num_electrons, 1))
        chex.assert_shape(nuclei.position, (self.num_nuclei, 3))
        chex.assert_shape(nuclei.charge, (self.num_nuclei, ))

        distances = calculate_distances(electrons.position, nuclei.position)
    
        @nnx.vmap(in_axes=(0, None, None, None), out_axes=(0, 0, 0))
        def calc_slater(slater: SlaterDeterminant, distances: Distances, spins: Array, distance_encoders: DistanceEncoders) -> Array:
            return slater(distances, spins, distance_encoders=distance_encoders)

        slater_logs, slater_signs, orthonormality_measures = calc_slater(self.slater_determinant, distances, electrons.spin, self.distance_encoders)
        slaters_sum, sign = slater_log_sum_exp(slater_logs, slater_signs, self.slater_strengths)

        result = self.jastrow_factor(distances, nuclei.charge, electrons.spin, self.distance_encoders) + slaters_sum

        return result.squeeze(), sign, jnp.mean(orthonormality_measures)  # Ensure we return a scalar

    def eval_wrt_electron_position(self, position: Array, nuclei: Nucleus, position_index: int, all_positions: Array, spins: Array) -> Array:
        """
        Helper function to evaluate the wave function with respect to a specific electron's position.
        This is used in the Hamiltonian to compute the kinetic term.
        """
        modified_positions = all_positions.at[position_index].set(position)
        electrons = Electron(position=modified_positions, spin=spins)
        return self(electrons, nuclei)


@nnx.jit(static_argnames=["sum_of_charges", "num_chains"])
def init_electrons(nuclei: Nucleus, num_chains: int, rng: Key, sum_of_charges: int) -> Electron:
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
    num_electrons = sum_of_charges
    positions = jnp.repeat(nuclei.position, nuclei.charge, axis=0, total_repeat_length=num_electrons)

    # Split the random key for position and spin generation.
    pos_rng, spin_rng = jax.random.split(rng)

    # Generate random positions around the nuclei for each chain.
    # The positions are created with a shape of (num_chains, num_electrons, 3).
    positions = jnp.expand_dims(positions, axis=0) + jax.random.normal(pos_rng, shape=(num_chains, num_electrons, 3))

    # For now the spins are not being changed in the MCMC walk, we will initialize it to -1, 1 in an alternating way
    spins = jnp.tile(((-1) ** jnp.arange(num_electrons))[None, :, None], (num_chains, 1, 1))
    # Generate random spins for each electron in each chain.
    # spins = jax.random.choice(spin_rng, jnp.array([-1, 1]), shape=(num_chains, num_electrons))

    # print(f"Positions shape: {positions.shape}, Spins shape: {spins.shape}")
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


def setup_sampler(
    wave_function: WaveFunction,
    nuclei: Nucleus,
    initial_positions: Electron,
    key: Key,
    num_chains: int,
    warmup_steps: int = 25,
):
    def call_wavefunction_wrapper(electron_positions: Array, electron_spins: Array, nuclei: Nucleus, graphdef: nnx.GraphDef, state: nnx.State) -> Array:
        wave_function = nnx.merge(graphdef, state)
        # print(f"Electrons shape: {electrons.position.shape}")
        electrons = Electron(position=electron_positions, spin=electron_spins)
        log_psi, _sign, _ortho_measure = wave_function(electrons, nuclei)
        # print(f"Wave function output shape: {log_psi.shape}")
        return 2 * log_psi

    graphdef, state = nnx.split(wave_function)

    initial_step_size = 0.25

    logdensity_fn = partial(call_wavefunction_wrapper,
                            electron_spins=initial_positions.spin[0],
                            nuclei=nuclei,
                            graphdef=graphdef,
                            state=state
                            )


    # warmup = blackjax.chees_adaptation(logdensity_fn, num_chains)
    # optim = optax.adam(1e-3)
    # (chain_state, parameters), _ = warmup.run(
    #     key,
    #     initial_positions.position,
    #     initial_step_size,
    #     optim,
    #     warmup_steps,
    # )
    # kernel = jax.vmap(blackjax.dynamic_hmc(logdensity_fn, **parameters).step, axis_size=num_chains)


    random_walk = blackjax.additive_step_random_walk(
        logdensity_fn,
        walk_electrons(initial_step_size)
    )
    chain_state = jax.vmap(random_walk.init)(initial_positions.position)
    kernel = jax.vmap(random_walk.step, axis_size=num_chains)


    # mala = blackjax.mala(logdensity_fn, step_size=0.1)
    # chain_state = jax.vmap(mala.init)(initial_positions.position)
    # kernel = jax.vmap(mala.step, axis_size=num_chains)

    return chain_state, kernel


def sample_from_wavefunction_using_sampler(chain_state, kernel, num_chains: int, num_samples: int, key: Key) -> ArrayLike:
    @nnx.scan(in_axes=(0, nnx.Carry), out_axes=(0, 0, nnx.Carry, 0), length=num_samples)
    def scan_step(step_key, chain_state):
        # new_chain_state, info = step(step_key, chain_state)
        sample_keys = jax.random.split(step_key, num_chains)
        new_chain_state, info = kernel(sample_keys, chain_state)
        # print(f"New chain state: {new_chain_state}")
        # electron_state_sample, log_density = new_chain_state
        electron_state_sample = new_chain_state.position
        log_density = new_chain_state.logdensity

        acceptance_rate = info.acceptance_rate

        return electron_state_sample, log_density, new_chain_state, acceptance_rate

    step_keys = jax.random.split(key, num_samples)
    electron_position_samples, log_densities, chain_state, acceptance_rate = scan_step(step_keys, chain_state)

    return electron_position_samples, log_densities, chain_state, jnp.mean(acceptance_rate)

def sample_from_wavefunction(
    wave_function: WaveFunction,
    nuclei: Nucleus,
    sum_of_charges: int,
    key: Key,
    num_chains: int,
    num_samples: int,
    warmup_steps: int = 100,
):
    init_electrons_key, init_sampler_key, sample_key = jax.random.split(key, 3)

    initial_positions = init_electrons(nuclei, num_chains=num_chains, rng=init_electrons_key, sum_of_charges=sum_of_charges)
    mcmc_chain_state, mcmc_kernel = setup_sampler(wave_function, nuclei, initial_positions, init_sampler_key, num_chains, warmup_steps)

    electron_position_samples, log_densities, mcmc_chain_state, acceptance_rate = sample_from_wavefunction_using_sampler(mcmc_chain_state, mcmc_kernel, num_chains, num_samples, sample_key)
    # print(f"Electron position shape: {electron_position_samples.shape}, initial position spin shape: {initial_positions.spin.shape}")
    # replicated_spins = jnp.tile(initial_positions.spin, (num_samples, 1, 1, 1))

    electron_position_samples = electron_position_samples[100::4, ...]
    replicated_spins = jnp.tile(initial_positions.spin, (electron_position_samples.shape[0], 1, 1, 1))

    return Electron(position=electron_position_samples, spin=replicated_spins), log_densities, mcmc_chain_state, acceptance_rate

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
def optimize_wave_function(wave_function: nnx.Module, optimizer: nnx.Optimizer, samples: Electron, nuclei: Nucleus, key: Key) -> Array:
    keys = jax.random.split(key, samples.position.shape[0])
    local_energies, log_grads, hamiltonian_details, log_psi = nnx.vmap(local_energy, in_axes=(None, 0, None, 0))(wave_function, samples, nuclei, keys)
    hamiltonian_details = jax.tree.map(lambda x: jnp.mean(x, axis=0), hamiltonian_details)
    # print(f"Log gradients: {log_grads}")
    
    avg_energy = jnp.mean(local_energies)
    centered_energies = local_energies - avg_energy
    # print(f"Energy offsets: {centered_energies}")

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
    samples, log_densities, _chain_state, acceptance_rate = sample_from_wavefunction(
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
    step_keys = jax.random.split(step_key, num_steps)
    for step, step_key in enumerate(step_keys):
        print(f"Step {step}/{num_steps}")

        samples, avg_energy, hamiltonian_details, log_psi, acceptance_rate = sample_and_optimize_wave_function(
            wave_function=wave_function,
            optimizer=optimizer,
            nuclei=nuclei,
            sum_of_charges=int(sum_of_charges),
            num_chains=int(num_chains),
            num_samples=int(num_samples),
            key=step_key,
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
    electronic_optimizer = nnx.Optimizer(
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
