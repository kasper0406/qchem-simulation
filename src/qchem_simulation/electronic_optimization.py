import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from jaxtyping import Array, Key, ArrayLike
from typing import List, Optional, Callable
import blackjax
from flax import nnx
import einops
from functools import partial
import optax
from .utils import Electron, Nucleus, clip_grad_elementwise, clip_grad_global, reduce_vmap
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


def calculate_kinetic_energy(wave_function: "WaveFunction", electrons: Electron, nuclei: Nucleus) -> Array:
    """
    Assumes wave_function returns the log-amplitude log(psi) of the wave function (returns both amplitude and sign).
    """
    from folx import forward_laplacian
    position_indices = jnp.arange(electrons.position.shape[0])

    @nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
    def calc_terms(wave_function: WaveFunction, position: Array, electron_idx: int) -> Array:
        # Evaluate the wave function with respect to the electron's position
        def calc_wrt_position(position: Array, graphdef: nnx.GraphDef, state: nnx.State) -> Array:
            wave_function = nnx.merge(graphdef, state)
            log_amplitude, _sign, _ortho_measure = wave_function.eval_wrt_electron_position(position, nuclei, electron_idx, electrons.position, electrons.spin)
            return log_amplitude

        graphdef, state = nnx.split(wave_function)
        laplacian = forward_laplacian(partial(calc_wrt_position, graphdef=graphdef, state=state))(position)
        # print(f"Jacobian: {amplitude_laplacian.jacobian}")
        # print(f"Laplacian: {amplitude_laplacian}")

        # Use the identity: ∇^2 psi = psi * (∇^2 log(psi) + |∇log(psi)|^2)
        # We actually want to calculate (∇^2 psi)/psi (the local energy, and from linearity of the Laplacian term
        # in the kinetic energy). Therefore, we calculate: (∇^2 psi)/psi = (∇^2 log(psi) + |∇log(psi)|^2)
        # where |∇log(psi)|^2 = J \dot J^T where J is the Jacobian of the wave function
        # Notice that since the laplacian is linear, the sign of psi will cancel out in (∇^2 psi)/psi.
        chex.assert_shape(laplacian.dense_jacobian, (3, ))
        jacobian_squared = jnp.sum(laplacian.dense_jacobian * laplacian.dense_jacobian, axis=-1)
        local_kinetic_term = laplacian.laplacian + jacobian_squared
        return local_kinetic_term

    terms = calc_terms(wave_function, electrons.position, position_indices)
    # print(f"Laplacian terms shape: {terms.shape}")
    return -0.5 * jnp.sum(terms)


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

    unit_vectors = diff * jax.lax.rsqrt(distances[:, :, None] + eps)

    if upper_half:
        # Keep only the upper half of the matrix (i.e., distances between different electrons)
        # print("Inside upper half calculation")
        upper_indices = jnp.triu_indices(distances.shape[0], k=1)
        distances = jnp.reshape(distances[upper_indices], (-1, ))
        square_distances = jnp.reshape(square_distances[upper_indices], (-1, ))

        # Notice the unit vectors will only keep one direction per pair of particles
        # when the upper half is selected
        unit_vectors = jnp.reshape(unit_vectors[upper_indices], (-1, 3))

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
        self.direction_proj = nnx.Linear(15, buckets, rngs=rngs)

        self.feature_up_proj = nnx.Linear(2 * buckets, hidden_dim, rngs=rngs)
        self.feature_down_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.feature_attn = nnx.MultiHeadAttention(num_heads=1, in_features=hidden_dim, decode=False, rngs=rngs)

        self.final_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def _rbf_encoding(self, magnitudes: Array):
        """
        Radial basis function (RBF) encoding for distance magnitudes.
        """
        chex.assert_shape(magnitudes, (None,))  # Expecting a 1D array of magnitudes
        return jnp.exp(-0.5 * ((magnitudes[:, None] - self.centers[None, :]) / self.sigma) ** 2)

    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def _real_sh_L3(self, direction: Array):
        chex.assert_shape(direction, (None, 3))  # Expecting a 3D unit vector
        x, y, z = jnp.moveaxis(direction, -1, 0)

        # l = 1 (dipoles)
        c1 = 0.4886025119029199
        Y1m1 = c1 * y
        Y10  = c1 * z
        Y11  = c1 * x

        # l = 2 (quadrupoles)
        Y2m2 = 1.0925484305920792 * x * y
        Y2m1 = 1.0925484305920792 * y * z
        Y20  = 0.31539156525252005 * (3.0 * z*z - 1.0)
        Y21  = 1.0925484305920792 * x * z
        Y22  = 0.5462742152960396 * (x*x - y*y)

        # l = 3 (octupoles) – fully-normalized real SH
        Y3m3 = 0.5900435899266435 * y * (3.0 * x*x - y*y)
        Y3m2 = 2.890611442640554 * x * y * z
        Y3m1 = 0.4570457994644658 * y * (5.0 * z*z - 1.0)
        Y30  = 0.3731763325901154 * z * (5.0 * z*z - 3.0)
        Y31  = 0.4570457994644658 * x * (5.0 * z*z - 1.0)
        Y32  = 1.445305721320277 * z * (x*x - y*y)
        Y33  = 0.5900435899266435 * x * (x*x - 3.0 * y*y)

        parts = [
            Y1m1, Y10, Y11,
            Y2m2, Y2m1, Y20, Y21, Y22,
            Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33
        ]
        stacked = jnp.stack(parts, axis=-1)
        chex.assert_shape(stacked, (None, 15))

        return stacked

    def __call__(self, distance: Distance):
        chex.assert_shape(distance.magnitude, (None, None))
        chex.assert_shape(distance.direction, (None, None, 3))

        radial_features = self.rbf_proj(self._rbf_encoding(distance.magnitude))
        direction_features = self.direction_proj(self._real_sh_L3(distance.direction))
        combined_features = jnp.concat([radial_features, direction_features], axis=-1)

        up_projected = self.feature_up_proj(combined_features)
        activated = nnx.gelu(up_projected)
        down_projected = self.feature_down_proj(activated)
        attended = self.feature_attn(down_projected)

        pooled_attention = jnp.mean(attended, axis=1)  # Mean to make sure the distance encoding will be permutation equivalent
        distance_encoding = self.final_proj(pooled_attention)

        chex.assert_shape(distance_encoding, (None, self.final_proj.out_features))

        return distance_encoding


class FlatDistanceEncoder(nnx.Module):
    """
    Use the ordinary DistanceEncoder, but we expect the argument to be a flat array of distances.
    We will add an initial dimension to make it compatible with the DistanceEncoder.
    """

    def __init__(self, hidden_dim: int, buckets: int, rngs: nnx.Rngs, cutoff: float = 10.0):
        self.encoder = DistanceEncoder(hidden_dim, buckets, rngs, cutoff)

    def __call__(self, distances: Distance):
        expanded = jax.tree.map(lambda x: x[None, :], distances)
        return self.encoder(expanded)[0]


@dataclass
class DistanceEncoders(nnx.Object):
    electron: DistanceEncoder
    electron_nuclei: DistanceEncoder
    nuclei: FlatDistanceEncoder


class ElectronicAttention(nnx.Module):
    def __init__(self, hdim: int, num_electrons: int, num_nuclei: int, rngs: nnx.Rngs):
        self.transformer = TransformerStack(
            num_layers=4,
            num_heads=2,
            hidden_dim=hdim,
            rngs=rngs,
        )

    def __call__(self, distances: Distances, distance_encoders: DistanceEncoders, extra_features: Array | None = None) -> Array:
        # print("Electron distances:", distances.electron_distances.shape)
        # print("Nuclei distances:", distances.nuclei_distances.shape)
        # print("Electron-nuclei distances:", distances.electron_nuclei_distances.shape)

        # In order to make the attention mechanism permutation-equivariant, we need to sort the distances
        # distances = jax.tree.map(jnp.sort, distances)

        electron_distance_embedding = distance_encoders.electron(distances.electron_distances)
        electron_nuclei_distance_embedding = distance_encoders.electron_nuclei(distances.electron_nuclei_distances)

        # The nuclei distance embedding is broadcasted over all electrons
        nuclei_distance_embedding = distance_encoders.nuclei(distances.nuclei_distances)[None, :]

        distance_embedding = electron_distance_embedding + electron_nuclei_distance_embedding + nuclei_distance_embedding
        print(f"Distance embedding shape: {distance_embedding.shape}")

        combined_embedding = distance_embedding
        if extra_features is not None:
            # print(f"Extra features shape: {extra_features.shape}")
            chex.assert_equal_shape([combined_embedding, extra_features])
            combined_embedding += extra_features

        print(f"Combined embedding shape: {combined_embedding.shape}")

        attention_output = self.transformer(combined_embedding)
        # attention_output = self.transformer(combined_embedding, electron_distance_embedding)

        return attention_output


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
            electron_nuclei=DistanceEncoder(hidden_dim, buckets=64, rngs=rngs),
            nuclei=FlatDistanceEncoder(hidden_dim, buckets=64, rngs=rngs),
        )

        self.jastrow_factor = JastrowFactor(num_electrons=num_electrons, num_nuclei=num_nuclei, hdim=hidden_dim, rngs=rngs)

        num_slaters = num_nuclei
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
    positions = jnp.repeat(nuclei.position, nuclei.charge.astype(jnp.int32), axis=0, total_repeat_length=num_electrons)

    # Split the random key for position and spin generation.
    pos_rng, spin_rng = jax.random.split(rng)

    # Generate random positions around the nuclei for each chain.
    # The positions are created with a shape of (num_chains, num_electrons, 3).
    sigma = 0.2  # Standard deviation for the normal distribution around each nucleus
    positions = jnp.expand_dims(positions, axis=0) + sigma * jax.random.normal(pos_rng, shape=(num_chains, num_electrons, 3))

    # For now the spins are not being changed in the MCMC walk, we will initialize it to -1, 1 in an alternating way
    spins = jnp.tile(((-1.0) ** jnp.arange(num_electrons))[None, :, None], (num_chains, 1, 1))
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


@nnx.jit(static_argnames=["sum_of_charges", "num_chains", "num_samples", "mcmc_burnin_steps", "mcmc_thinning_factor"])
def sample_from_wavefunction(
    wave_function: WaveFunction,
    nuclei: Nucleus,
    sum_of_charges: int,
    key: Key,
    num_chains: int,
    num_samples: int,
    mcmc_burnin_steps: int = 50,
    mcmc_thinning_factor: int = 2,
):
    init_electrons_key, init_sampler_key, sample_key = jax.random.split(key, 3)

    initial_positions = init_electrons(nuclei, num_chains=num_chains, rng=init_electrons_key, sum_of_charges=sum_of_charges)
    chex.assert_shape(initial_positions.position, (num_chains, sum_of_charges, 3))
    num_electrons = initial_positions.position.shape[1]


    # Code the actually do MCMC sampling
    def call_wavefunction_wrapper(
        # The MCMC chain will move only the position of one electron at a time
        proposed_position_i: Array, # Shape (3,)
        electron_i: int,
        original_positions: Array, # Shape (num_electrons, 3)
        *,
        electron_spins: Array,
        nuclei: Nucleus,
        graphdef: nnx.GraphDef,
        state: nnx.State
    ) -> Array:
        chex.assert_shape(proposed_position_i, (3,))
        chex.assert_shape(original_positions, (num_electrons, 3))

        wave_function = nnx.merge(graphdef, state)
        proposed_positions = original_positions.at[electron_i].set(proposed_position_i)
        # print(f"Electrons shape: {all_positions.shape}")

        electrons = Electron(position=proposed_positions, spin=electron_spins)
        log_psi, _sign, _ortho_measure = wave_function(electrons, nuclei)
        # print(f"Wave function output shape: {log_psi.shape}")
        return 2 * log_psi

    graphdef, state = nnx.split(wave_function)
    logdensity_fn = partial(call_wavefunction_wrapper,
                            electron_spins=initial_positions.spin[0],
                            graphdef=graphdef,
                            nuclei=nuclei,
                            state=state
                            )

    mala_kernel = blackjax.mala.build_kernel()
    step_size = 0.1  # TODO: Figure out a good step size adaptation scheme

    def per_electron_kernel(
        rng_key: Key,
        electron_i: Array, # Scalar, which electron to update
        electron_positions: Array, # Shape (num_electrons, 3)
    ):
        # The gradient of the wave-functions tends to infinity around the cusps and nodes.
        # Clip the gradient during sampling to make it more stable.
        @clip_grad_global(max_norm=1.0)
        def per_electron_logdensity(
            proposed_position_i: Array, # Shape (3,)
        ) -> Array:
            return logdensity_fn(proposed_position_i, electron_i, electron_positions)
        chain_state = blackjax.mala.init(electron_positions[electron_i], logdensity_fn=per_electron_logdensity)

        chain_state, info = mala_kernel(rng_key, chain_state, per_electron_logdensity, step_size)

        acceptance_rate = info.acceptance_rate
        updated_electron_i_pos = chain_state.position
        log_density = chain_state.logdensity
        new_electron_positions = electron_positions.at[electron_i].set(updated_electron_i_pos)

        return new_electron_positions, log_density, acceptance_rate


    @nnx.scan(in_axes=(0, nnx.Carry, 0), out_axes=(0, nnx.Carry, 0, 0), length=num_samples)
    def scan_num_samples(
        electron_indices: Array, # Shape (num_samples, num_chains, num_electrons) the order in which to update the electrons
        electron_positions: Array, # Carry of shape (num_chains, num_electrons, 3)
        step_key: Key
    ):

        @jax.vmap
        def map_chains(
            electron_indices: Array, # Shape (num_chains, num_electrons)
            electron_positions: Array, # Shape (num_chains, num_electrons, 3)
            step_key: Key,
        ):

            # We want to update the electron positions one by one - proposing moves of all of them
            # at the same time, makes the acceptance rate very low and can cause weird behaviors
            @nnx.scan(in_axes=(0, nnx.Carry, 0), out_axes=(nnx.Carry, 0, 0))
            def scan_electrons(
                electron_i: Array, # Scalar, which electron to update on which chain
                electron_positions: Array, # Shape (num_electrons, 3): current positions of all electrons on this chain
                electron_step_key: Key,
            ):
                new_electron_positions, log_density, acceptance_rate = per_electron_kernel(electron_step_key, electron_i, electron_positions)
                # print(f"New chain state: {new_chain_state}")
                # electron_state_sample, log_density = new_chain_state

                return new_electron_positions, log_density, acceptance_rate

            new_electron_positions, log_density, acceptance_rate = scan_electrons(
                electron_indices,
                electron_positions,
                jax.random.split(step_key, num_electrons),
            )

            return new_electron_positions, jnp.mean(log_density), jnp.mean(acceptance_rate)


        chex.assert_shape(electron_indices, (num_chains, num_electrons))
        chex.assert_shape(electron_positions, (num_chains, num_electrons, 3))
        chain_keys = jax.random.split(step_key, num_chains)
        new_electron_positions, log_density, acceptance_rate = map_chains(
            electron_indices,
            electron_positions,
            chain_keys,
        )

        return new_electron_positions, new_electron_positions, log_density, acceptance_rate

    # We want to update each electron in a turn, but we do not always want to update in the order 0, 1, 2, ...
    # We will generate electron_permutations of shape (num_samples, num_chains, num_electrons) where there will be a
    # random permutation of the electrons for each sample step and chain
    # TODO: Refactor this to a utility function?
    @nnx.vmap(axis_size=num_samples)
    def generate_electron_permutation(key: Key) -> Array:
        @nnx.vmap(axis_size=num_chains)
        def _generate(key: Key) -> Array:
            return jax.random.permutation(key, jnp.arange(num_electrons))
        return _generate(jax.random.split(key, num_chains))
    electron_permutations = generate_electron_permutation(jax.random.split(sample_key, num_samples))
    chex.assert_shape(electron_permutations, (num_samples, num_chains, num_electrons))

    step_keys = jax.random.split(key, num_samples)
    electron_position_samples, _last_position, log_densities, acceptance_rate = scan_num_samples(electron_permutations, initial_positions.position, step_keys)
    chex.assert_shape(electron_position_samples, (num_samples, num_chains, num_electrons, 3))
    # print(f"Electron position shape: {electron_position_samples.shape}, initial position spin shape: {initial_positions.spin.shape}")
    # replicated_spins = jnp.tile(initial_positions.spin, (num_samples, 1, 1, 1))

    # Apply burn-in and thinning
    electron_position_samples = electron_position_samples[mcmc_burnin_steps::mcmc_thinning_factor, ...]
    replicated_spins = jnp.tile(initial_positions.spin, (electron_position_samples.shape[0], 1, 1, 1))

    chex.assert_shape(replicated_spins, (electron_position_samples.shape[0], num_chains, num_electrons, 1))
    return Electron(position=electron_position_samples, spin=replicated_spins), log_densities, jnp.mean(acceptance_rate)

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
    *,
    lambda_virial: float = 1e-3,
key: Key) -> Array:
    keys = jax.random.split(key, samples.position.shape[0])

    local_energies, log_grads, hamiltonian_details, log_psi = nnx.vmap(local_energy, in_axes=(None, 0, None, 0))(wave_function, samples, nuclei, keys)
    # print(f"Log gradients: {log_grads}")

    avg_energy = jnp.mean(local_energies)
    centered_energies = local_energies - avg_energy
    # print(f"Energy offsets: {centered_energies}")

    # The introduced penalty terms is to:
    #  1. keep 2T+V=0 (virial theorem)
    #  2. keep the orbitals orthonormal
    virial_penalty = jnp.square(2 * hamiltonian_details.kinetic_energy + hamiltonian_details.potential_energy)
    penalty_terms = lambda_virial * virial_penalty # + lambda_ortho * orthonormality_measure

    loss_gradient = jax.tree.map(
        lambda log_grads: jnp.expand_dims(centered_energies + penalty_terms, axis=tuple(range(1, log_grads.ndim))) * log_grads,
        log_grads,
    )
    # print(f"Loss gradient: {loss_gradient}")

    mean_loss_grads = jax.tree.map(
        lambda loss_gradient: jnp.mean(loss_gradient, axis=0),
        loss_gradient,
    )
    # print(f"Mean loss gradients: {mean_loss_grads}")

    optimizer.update(mean_loss_grads)

    mean_hamiltonian_details = jax.tree.map(lambda x: jnp.mean(x, axis=0), hamiltonian_details)
    return avg_energy, mean_hamiltonian_details, jnp.mean(log_psi)

@nnx.jit
def optimize_wave_function_batched(
    wave_function: WaveFunction,
    optimizer: nnx.Optimizer,
    samples: Electron,
    nuclei: Nucleus,
    key: Key,
    *,
    batch_size: int = 100,
    lambda_virial: float = 1e-3,
) -> Array:

    @nnx.scan(
        in_axes=(None, nnx.Carry, 0, 0),
        out_axes=nnx.Carry,
    )
    def process_batch(wave_function: WaveFunction, carry, batch_samples: Electron, batch_key):
        weighed_grads_sum, local_energy_sum, loss_term_sum, grads_sum, hamiltonian_details_sum, log_psi_sum = carry
        keys = jax.random.split(batch_key, batch_samples.position.shape[0])

        local_energies, log_grads, hamiltonian_details, log_psi = nnx.vmap(local_energy, in_axes=(None, 0, None, 0))(wave_function, batch_samples, nuclei, keys)

        # The introduced penalty terms is to:
        #  1. keep 2T+V=0 (virial theorem)
        #  2. keep the orbitals orthonormal
        virial_penalty = jnp.square(2 * hamiltonian_details.kinetic_energy + hamiltonian_details.potential_energy)
        penalty_terms = lambda_virial * virial_penalty # + lambda_ortho * orthonormality_measure

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
    #   E[((E_L - <E_L>) + penalty_terms) * log_grads] = E[E_L * log_grads] - <E_L> * E[log_grads]
    # Then because we take the sum, we get the average by dividing by the number of samples
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
