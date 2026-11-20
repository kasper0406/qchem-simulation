import chex
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qchem_simulation.electronic_optimization import (
    DistanceEncoder,
    calculate_pairwise_distances,
)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 4.0, 100.0])
def test_unit_vectors_have_unit_norm_across_distances(scale: float):
    pos_a = jnp.array([[0.0, 0.0, 0.0], [scale, 0.0, 0.0]], dtype=jnp.float32)
    pos_b = jnp.array(
        [
            [0.0, scale, 0.0],
            [scale, scale, scale],
            [-0.5 * scale, 0.25 * scale, -0.75 * scale],
        ],
        dtype=jnp.float32,
    )

    dist = calculate_pairwise_distances(pos_a, pos_b)
    chex.assert_shape(dist.magnitude, (2, 3))
    chex.assert_shape(dist.direction, (2, 3, 3))

    norms = jnp.linalg.norm(dist.direction, axis=-1)
    chex.assert_trees_all_close(norms, jnp.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_unit_vectors_point_in_exact_direction():
    pos_a = jnp.array([[1.0, -2.0, 3.5], [-4.0, 0.5, 2.0]], dtype=jnp.float32)
    pos_b = jnp.array([[0.0, 1.0, -1.5], [2.5, -3.5, 0.0], [-1.0, -1.0, 4.0]], dtype=jnp.float32)

    dist = calculate_pairwise_distances(pos_a, pos_b)

    diff = pos_a[:, None, :] - pos_b[None, :, :]
    expected_mag = jnp.linalg.norm(diff, axis=-1)
    expected_dir = diff / expected_mag[:, :, None]

    chex.assert_trees_all_close(dist.magnitude, expected_mag, atol=1e-5, rtol=1e-5)
    chex.assert_trees_all_close(dist.direction, expected_dir, atol=1e-5, rtol=1e-5)


def test_self_distances_have_zero_direction_and_finite_gradients():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, -2.0, 0.5],
            [-0.5, 3.0, 4.0],
        ],
        dtype=jnp.float32,
    )

    dist = calculate_pairwise_distances(positions, positions)

    # Diagonal entries correspond to self-distances (position_a == position_b)
    diag_dir = jnp.diagonal(dist.direction, axis1=0, axis2=1).T
    chex.assert_tree_all_finite(dist.magnitude)
    chex.assert_tree_all_finite(dist.direction)
    chex.assert_trees_all_close(diag_dir, jnp.zeros_like(diag_dir), atol=1e-7)

    # Off-diagonal entries have unit norm
    norms = jnp.linalg.norm(dist.direction, axis=-1)
    off_diag_mask = ~jnp.eye(positions.shape[0], dtype=bool)
    chex.assert_trees_all_close(norms[off_diag_mask], jnp.ones(6, dtype=jnp.float32), atol=1e-5)

    def loss_fn(pos):
        d = calculate_pairwise_distances(pos, pos)
        return jnp.sum(d.magnitude) + jnp.sum(d.direction) + jnp.sum(d.direction ** 2)

    grads = jax.grad(loss_fn)(positions)
    chex.assert_tree_all_finite(grads)


def test_upper_half_shapes_and_unit_norms():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 100.0],
        ],
        dtype=jnp.float32,
    )
    n = positions.shape[0]
    expected_pairs = n * (n - 1) // 2

    dist = calculate_pairwise_distances(positions, positions, upper_half=True)
    chex.assert_shape(dist.magnitude, (expected_pairs,))
    chex.assert_shape(dist.direction, (expected_pairs, 3))

    norms = jnp.linalg.norm(dist.direction, axis=-1)
    chex.assert_trees_all_close(norms, jnp.ones((expected_pairs,), dtype=jnp.float32), atol=1e-5)

    # Verify the directions match the upper-triangle pairs (i < j)
    full_dist = calculate_pairwise_distances(positions, positions, upper_half=False)
    iu = jnp.triu_indices(n, k=1)
    chex.assert_trees_all_close(dist.magnitude, full_dist.magnitude[iu], atol=1e-6)
    chex.assert_trees_all_close(dist.direction, full_dist.direction[iu], atol=1e-6)


def test_distance_encoder_bounded_at_large_distances():
    rngs = nnx.Rngs(42)
    encoder = DistanceEncoder(hidden_dim=16, buckets=32, rngs=rngs, cutoff=10.0)

    base_a = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]], dtype=jnp.float32)
    base_b = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=jnp.float32)

    # Evaluate at moderate (r ~ 1e2) and very large (r ~ 1e6) scales well beyond RBF cutoff.
    # Because RBF decays to 0 and unit vectors are scale-invariant, outputs must remain finite
    # and converge to the same bounded value rather than diverging as r^(3/2).
    dist_mid = calculate_pairwise_distances(base_a * 1e2, base_b * 1e2)
    dist_far = calculate_pairwise_distances(base_a * 1e6, base_b * 1e6)

    out_mid = encoder(dist_mid)
    out_far = encoder(dist_far)

    chex.assert_tree_all_finite(out_mid)
    chex.assert_tree_all_finite(out_far)
    chex.assert_trees_all_close(out_mid, out_far, atol=1e-4, rtol=1e-4)
