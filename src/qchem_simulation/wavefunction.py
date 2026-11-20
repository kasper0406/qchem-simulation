"""Neural Slater–Jastrow wavefunctions and their geometric features."""

from dataclasses import dataclass
from typing import List, Optional

import chex
from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .utils import Electron, Nucleus


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


class FeedForward(nnx.Module):
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        self.up_project = nnx.Linear(hidden_dim, hidden_dim * 4, rngs=rngs)
        self.down_project = nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = self.up_project(x)
        x = jax.nn.glu(x)
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
        self.layers = nnx.List([
            TransformerLayer(num_heads=num_heads, hidden_dim=hidden_dim, rngs=rngs)
            for _ in range(num_layers)
        ])

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

    unit_vectors = diff * jax.lax.rsqrt(square_distances[:, :, None] + eps)

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


class StaticValue(nnx.Variable):
    pass


class DistanceEncoder(nnx.Module):
    def __init__(self, hidden_dim: int, buckets: int, rngs: nnx.Rngs, cutoff: float = 10.0):
        self.centers = StaticValue(jnp.linspace(0.0, cutoff, buckets))
        self.sigma = StaticValue((self.centers[1] -  self.centers[0]) * 1.5)

        self.rbf_proj = nnx.Linear(buckets, buckets, rngs=rngs)
        self.direction_proj = nnx.Linear(15, buckets, rngs=rngs)

        self.feature_up_proj = nnx.Linear(2 * buckets, hidden_dim, rngs=rngs)
        self.feature_down_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.feature_attn = nnx.MultiHeadAttention(num_heads=1, in_features=hidden_dim, decode=False, rngs=rngs)

        self.final_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

    def __call__(self, distance: Distance):
        chex.assert_shape(distance.magnitude, (None, None))
        chex.assert_shape(distance.direction, (None, None, 3))

        def _rbf_encoding(magnitudes: Array):
            """
            Radial basis function (RBF) encoding for distance magnitudes.
            """
            chex.assert_shape(magnitudes, (None, None))  # Expecting a magnitude for every pair
            diffs = (magnitudes[:, :, None] - self.centers[None, None, :]) / self.sigma
            return jnp.exp(-0.5 * (diffs ** 2))

        @nnx.vmap
        def _real_sh_L3(direction: Array):
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

        radial_features = self.rbf_proj(_rbf_encoding(distance.magnitude))
        direction_features = self.direction_proj(_real_sh_L3(distance.direction))
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
        if distances.magnitude.shape[0] == 0:
            return jnp.zeros((self.encoder.final_proj.out_features,), dtype=distances.magnitude.dtype)
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


def spin_up_count(num_electrons: int, num_up: Optional[int]) -> int:
    """Resolve the fixed spin population shared by the model and sampler."""
    if num_up is None:
        return (num_electrons + 1) // 2
    if not (0 <= num_up <= num_electrons):
        raise ValueError(f"num_up ({num_up}) must be between 0 and num_electrons ({num_electrons}).")
    return num_up


class OrbitalNetwork(nnx.Module):
    def __init__(
        self,
        hdim: int,
        num_electrons: int,
        num_nuclei: int,
        rngs: nnx.Rngs,
        num_up: Optional[int] = None,
    ):
        num_up = spin_up_count(num_electrons, num_up)

        self.num_electrons = num_electrons
        self.num_up = num_up
        self.num_down = num_electrons - num_up

        self.spin_encoder = nnx.Embed(2, features=hdim, rngs=rngs)
        self.electronic_attention = ElectronicAttention(hdim, num_electrons, num_nuclei, rngs=rngs)

        self.up_head = OrbitalHead(hdim, self.num_up, rngs=rngs) if self.num_up > 0 else None
        self.down_head = OrbitalHead(hdim, self.num_down, rngs=rngs) if self.num_down > 0 else None

    def __call__(
        self, distances: Distances, spins: Array, distance_encoders: DistanceEncoders
    ) -> tuple[tuple[Optional[Array], Optional[Array]], Array]:
        chex.assert_shape(spins, (self.num_electrons, 1))  # Expecting spins to be a column vector

        spin_classes = jnp.where(spins > 0, 1, 0)
        spin_embedding = jnp.squeeze(self.spin_encoder(spin_classes), axis=-2)

        attention = self.electronic_attention(
            distances, distance_encoders=distance_encoders, extra_features=spin_embedding
        )

        is_up = spins[:, 0] > 0
        orthonormality_measure = jnp.zeros((), dtype=attention.dtype)

        up_orbitals: Optional[Array] = None
        if self.num_up > 0:
            up_indices = jnp.nonzero(is_up, size=self.num_up)[0]
            up_orbitals = self.up_head(attention[up_indices])
            up_gram = up_orbitals.T @ up_orbitals - jnp.eye(self.num_up, dtype=up_orbitals.dtype)
            orthonormality_measure = orthonormality_measure + jnp.sum(up_gram ** 2, axis=(-1, -2))

        down_orbitals: Optional[Array] = None
        if self.num_down > 0:
            down_indices = jnp.nonzero(~is_up, size=self.num_down)[0]
            down_orbitals = self.down_head(attention[down_indices])
            down_gram = down_orbitals.T @ down_orbitals - jnp.eye(self.num_down, dtype=down_orbitals.dtype)
            orthonormality_measure = orthonormality_measure + jnp.sum(down_gram ** 2, axis=(-1, -2))

        return (up_orbitals, down_orbitals), orthonormality_measure


class SlaterDeterminant(nnx.Module):
    def __init__(
        self,
        num_electrons: int,
        num_nuclei: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        num_up: Optional[int] = None,
    ):
        self.orbital_network = OrbitalNetwork(
            hidden_dim, num_electrons, num_nuclei, rngs=rngs, num_up=num_up
        )

    def __call__(
        self, distances: Distances, spins: Array, distance_encoders: DistanceEncoders
    ) -> tuple[Array, Array, Array]:
        (up_orbitals, down_orbitals), orthonormality_measure = self.orbital_network(
            distances, spins, distance_encoders=distance_encoders
        )

        if up_orbitals is not None:
            dtype = up_orbitals.dtype
        elif down_orbitals is not None:
            dtype = down_orbitals.dtype
        else:
            dtype = jnp.float32

        total_logabsdet = jnp.zeros((), dtype=dtype)
        # Regrouping labeled electrons into up and down blocks permutes rows.
        # Each down electron preceding an up electron contributes one inversion.
        is_up = spins[:, 0] > 0
        inversions = jnp.sum(jnp.where(is_up, jnp.cumsum(~is_up), 0))
        total_sign = jnp.where(inversions % 2 == 0, 1, -1).astype(dtype)

        if up_orbitals is not None:
            up_det = jnp.linalg.slogdet(up_orbitals, method="qr")
            total_logabsdet = total_logabsdet + up_det.logabsdet
            total_sign = total_sign * up_det.sign

        if down_orbitals is not None:
            down_det = jnp.linalg.slogdet(down_orbitals, method="qr")
            total_logabsdet = total_logabsdet + down_det.logabsdet
            total_sign = total_sign * down_det.sign

        # A fixed-sector state has zero amplitude for every other spin population.
        in_sector = jnp.sum(is_up) == self.orbital_network.num_up
        return (
            jnp.where(in_sector, total_logabsdet, -jnp.inf),
            jnp.where(in_sector, total_sign, 0),
            orthonormality_measure,
        )


class CuspElectrons(nnx.Module):
    """
    Models the Kato cusp condition (https://en.wikipedia.org/wiki/Kato_theorem) for electrons.
    """
    def __init__(self, num_electrons: int):
        electron_pairs = (num_electrons * (num_electrons - 1)) // 2
        if electron_pairs > 0:
            self.decay_strength_parallel = nnx.Param(jnp.array(1.0))
            self.decay_strength_antiparallel = nnx.Param(jnp.array(1.0))
        else:
            self.decay_strength_parallel = None
            self.decay_strength_antiparallel = None

    def __call__(self, electron_distances: Array, spins: Array) -> Array:
        chex.assert_rank(spins, 2)
        chex.assert_axis_dimension(spins, 1, 1)
        chex.assert_axis_dimension(spins, 0, electron_distances.shape[0])
        chex.assert_shape(electron_distances, (spins.shape[0], spins.shape[0]))

        if self.decay_strength_parallel is None:
            return 0.0

        upper_indices = jnp.triu_indices(electron_distances.shape[0], k=1)

        flat_pairwise_distances = jnp.reshape(electron_distances[upper_indices], (-1, ))

        spins = jnp.squeeze(spins, axis=-1)
        spin_correlations = spins[:, None] == spins[None, :]
        flat_spin_correlations = jnp.reshape(spin_correlations[upper_indices], (-1, ))

        flat_cusp_values = jnp.where(flat_spin_correlations, 0.25, 0.5)
        decay_parallel = jax.nn.softplus(self.decay_strength_parallel[...])
        decay_antiparallel = jax.nn.softplus(self.decay_strength_antiparallel[...])
        flat_decay_strengths = jnp.where(flat_spin_correlations, decay_parallel, decay_antiparallel)

        # log-Jastrow factor on the form: \sum_{i < j} c_{ij} * d_{ij} / (1 + b_{ij} * d_{ij})
        # where b_{ij} > 0 depends only on whether spins are parallel or anti-parallel.
        jastrow_factor = jnp.sum(
            (flat_cusp_values * flat_pairwise_distances) / (1.0 + flat_decay_strengths * flat_pairwise_distances),
            axis=(-1,),
        )
        return jastrow_factor


class CuspElectronNuclei(nnx.Module):
    """
    Models the Kato cusp condition (https://en.wikipedia.org/wiki/Kato_theorem) for electron-nucleus pairs.
    """
    def __init__(self, num_electrons: int, num_nuclei: int):
        self.num_electrons = num_electrons
        self.num_nuclei = num_nuclei

        self.decay_strengths = nnx.Param(jnp.ones((num_nuclei,)))

    def __call__(self, electron_nuclei_distances: Array, nuclei_charges: Array):
        chex.assert_shape(electron_nuclei_distances, (self.num_electrons, self.num_nuclei))
        chex.assert_shape(nuclei_charges, (self.num_nuclei,))

        decay_strengths = jax.nn.softplus(self.decay_strengths[...])

        # log-Jastrow factor on the form: \sum_{i, I} -Z_I * d_{iI} / (1 + b_I * d_{iI})
        # where b_I > 0 is the nucleus-specific decay strength broadcast over all electrons.
        jastrow_factor = -jnp.sum(
            (nuclei_charges[None, :] * electron_nuclei_distances)
            / (1.0 + decay_strengths[None, :] * electron_nuclei_distances),
            axis=(-1, -2),
        )
        return jastrow_factor


class ExponentialEnvelope(nnx.Module):
    """
    Permutation-invariant learnable exponential decay envelope ensuring normalizability:
    log_env(r) = sum_i log(sum_I pi_I * exp(-sigma_I * r_{iI}))
    where pi_I > 0 and sigma_I > 0 are per-nucleus parameters broadcast over all electrons.
    """
    def __init__(self, num_nuclei: int):
        self.num_nuclei = num_nuclei
        self.pi = nnx.Param(jnp.ones((num_nuclei,)))
        self.sigma = nnx.Param(jnp.ones((num_nuclei,)))

    def __call__(self, electron_nuclei_distances: Array) -> Array:
        chex.assert_rank(electron_nuclei_distances, 2)
        chex.assert_axis_dimension(electron_nuclei_distances, 1, self.num_nuclei)

        pi = jax.nn.softplus(self.pi[...]) + 1e-8
        sigma = jax.nn.softplus(self.sigma[...])

        log_terms = jnp.log(pi[None, :]) - sigma[None, :] * electron_nuclei_distances
        return jnp.sum(jax.scipy.special.logsumexp(log_terms, axis=-1), axis=-1)


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
    slater_logs = jnp.asarray(slater_logs)
    slater_signs = jnp.asarray(slater_signs)
    slater_coeffs = jnp.asarray(slater_coeffs)

    dtype = jnp.result_type(slater_logs, slater_coeffs, jnp.float32)
    eps = jnp.asarray(1e-12, dtype=dtype)

    abs_coeffs = jnp.abs(slater_coeffs)
    valid_mask = (abs_coeffs > 0) & jnp.isfinite(slater_logs)
    safe_abs_coeffs = jnp.where(valid_mask, abs_coeffs, jnp.ones_like(abs_coeffs))
    safe_logs = jnp.where(valid_mask, slater_logs, jnp.zeros_like(slater_logs))

    # Calculate the log-magnitude of each term in the expansion
    # log|T_i| = log|c_i| + log|D_i|
    log_term_magnitudes = jnp.log(safe_abs_coeffs) + safe_logs

    # Find the dominant term's magnitude among valid terms (the 'x_max' in the standard trick)
    any_valid = jnp.any(valid_mask)
    min_log_mag = jnp.min(log_term_magnitudes) - jnp.asarray(1.0, dtype=dtype)
    masked_log_mags = jnp.where(valid_mask, log_term_magnitudes, min_log_mag)
    max_log_mag = jnp.where(any_valid, jnp.max(masked_log_mags), jnp.asarray(0.0, dtype=dtype))
    max_log_mag = jax.lax.stop_gradient(max_log_mag)

    # Calculate the sum of terms, normalized by the largest term to prevent overflow.
    # The value of each term in the sum is:
    # sign(c_i)*s_i*exp(log|c_i| + ld_i - max_log_mag)
    scaled_exponents = jnp.where(valid_mask, log_term_magnitudes - max_log_mag, jnp.asarray(0.0, dtype=dtype))
    terms = jnp.where(
        valid_mask,
        jnp.sign(slater_coeffs) * slater_signs * jnp.exp(scaled_exponents),
        jnp.asarray(0.0, dtype=dtype),
    )
    sum_val = jnp.sum(terms)

    # The final sign is the sign of this normalized sum
    sign_psi = jnp.sign(sum_val)

    # The final log-magnitude is the log of the dominant term's magnitude
    # plus the log of the absolute value of the normalized sum.
    # Clamp |sum_val| away from 0 so nodal surfaces (sum_val == 0) yield finite values and gradients.
    safe_abs_sum = jnp.maximum(jnp.abs(sum_val), eps)
    log_psi_abs = max_log_mag + jnp.log(safe_abs_sum)

    return log_psi_abs, sign_psi


class WaveFunction(nnx.Module):
    def __init__(
        self,
        num_electrons: int,
        num_nuclei: int,
        rngs: nnx.Rngs,
        num_up: Optional[int] = None,
    ):
        num_up = spin_up_count(num_electrons, num_up)

        self.num_electrons = num_electrons
        self.num_nuclei = num_nuclei
        self.num_up = num_up
        self.num_down = num_electrons - num_up

        hidden_dim = 64
        self.distance_encoders = DistanceEncoders(
            electron=DistanceEncoder(hidden_dim, buckets=64, rngs=rngs),
            electron_nuclei=DistanceEncoder(hidden_dim, buckets=64, rngs=rngs),
            nuclei=FlatDistanceEncoder(hidden_dim, buckets=64, rngs=rngs),
        )

        self.jastrow_factor = JastrowFactor(num_electrons=num_electrons, num_nuclei=num_nuclei, hdim=hidden_dim, rngs=rngs)
        self.envelope = ExponentialEnvelope(num_nuclei=num_nuclei)

        num_slaters = num_nuclei
        # @nnx.split_rngs(splits=num_slaters)
        # @nnx.vmap
        # def create_slater_determinants(rng: Key) -> SlaterDeterminant:
        #     return SlaterDeterminant(num_electrons, num_nuclei, hidden_dim, rngs=rng)
        # self.slater_determinants = create_slater_determinants(rngs)
        self.slater_determinants = nnx.List([
            SlaterDeterminant(num_electrons, num_nuclei, hidden_dim, rngs=rngs, num_up=self.num_up)
            for _ in range(num_slaters)
        ])

        self.slater_strengths = nnx.Param(jnp.ones((num_slaters,), dtype=jnp.float32))

    def __call__(self, electrons: Electron, nuclei: Nucleus) -> Array:
        chex.assert_shape(electrons.position, (self.num_electrons, 3))
        chex.assert_shape(electrons.spin, (self.num_electrons, 1))
        chex.assert_shape(nuclei.position, (self.num_nuclei, 3))
        chex.assert_shape(nuclei.charge, (self.num_nuclei, ))

        distances = calculate_distances(electrons.position, nuclei.position)

        # @nnx.vmap(in_axes=(0, None, None, None), out_axes=(0, 0, 0))
        # def calc_slater(slater: SlaterDeterminant, distances: Distances, spins: Array, distance_encoders: DistanceEncoders) -> Array:
        #     return slater(distances, spins, distance_encoders=distance_encoders)
        def calc_slater(slaters: List[SlaterDeterminant], distances: Distances, spins: Array, distance_encoders: DistanceEncoders) -> Array:
            results = [slater(distances, spins, distance_encoders=distance_encoders) for slater in slaters]
            slater_logs, slater_signs, orthonormality_measures = zip(*results)
            # print(f"Slater logs: {slater_logs}")
            # print(f"Slater signs: {slater_signs}")
            # print(f"Orthonormality measures: {orthonormality_measures}")
            return jnp.array(slater_logs), jnp.array(slater_signs), jnp.array(orthonormality_measures)

        slater_logs, slater_signs, orthonormality_measures = calc_slater(self.slater_determinants, distances, electrons.spin, self.distance_encoders)
        slaters_sum, sign = slater_log_sum_exp(slater_logs, slater_signs, self.slater_strengths)

        envelope = self.envelope(distances.electron_nuclei_distances.magnitude)
        result = self.jastrow_factor(distances, nuclei.charge, electrons.spin, self.distance_encoders) + slaters_sum + envelope

        # The signed-log reducer regularizes spatial nodes; an excluded spin
        # sector is exactly zero independent of the electron positions.
        in_sector = jnp.sum(electrons.spin > 0) == self.num_up
        return jnp.where(in_sector, result.squeeze(), -jnp.inf), sign, jnp.mean(orthonormality_measures)

    def eval_wrt_electron_position(self, position: Array, nuclei: Nucleus, position_index: int, all_positions: Array, spins: Array) -> Array:
        """
        Helper function to evaluate the wave function with respect to a specific electron's position.
        This is used in the Hamiltonian to compute the kinetic term.
        """
        modified_positions = all_positions.at[position_index].set(position)
        electrons = Electron(position=modified_positions, spin=spins)
        return self(electrons, nuclei)
