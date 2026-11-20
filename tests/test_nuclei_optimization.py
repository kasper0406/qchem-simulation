import chex
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from qchem_simulation.nuclei_optimization import (
    nuclei_energy_grad_full,
    nuclei_energy_grad_hellmann_feynman,
    sample_and_optimize,
    simplified_hamiltonian,
)
from qchem_simulation.utils import Electron, Nucleus


class _GaussianWaveFunction(nnx.Module):
    """Smooth analytical wave function for end-to-end sampling/optimization tests."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = nnx.Param(jnp.array(alpha, dtype=jnp.float32))

    def __call__(self, electrons: Electron, nuclei: Nucleus):
        log_psi = -self.alpha.value * jnp.sum(electrons.position ** 2)
        return log_psi, jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)


def _exact_coulomb_grad(
    electron_positions: jax.Array,
    nuclei_positions: jax.Array,
    nuclei_charges: jax.Array,
) -> jax.Array:
    """Closed-form gradient dV/dR_I of the Coulomb potential for one electron configuration.

    V(R) = sum_{I < J} Z_I Z_J / |R_I - R_J| - sum_{i, I} Z_I / |r_i - R_I|
    dV/dR_I = -sum_{J != I} Z_I Z_J (R_I - R_J) / |R_I - R_J|^3
              + sum_i Z_I (R_I - r_i) / |R_I - r_i|^3
    """
    num_nuclei = nuclei_positions.shape[0]
    grads = []
    for i in range(num_nuclei):
        r_i = nuclei_positions[i]
        z_i = nuclei_charges[i]

        # Nuclear-nuclear repulsion gradient
        nn_grad = jnp.zeros(3, dtype=nuclei_positions.dtype)
        for j in range(num_nuclei):
            if i == j:
                continue
            diff = r_i - nuclei_positions[j]
            dist = jnp.linalg.norm(diff)
            nn_grad = nn_grad - z_i * nuclei_charges[j] * diff / (dist ** 3)

        # Electron-nuclear attraction gradient
        diff_e = r_i[None, :] - electron_positions  # (num_electrons, 3)
        dist_e = jnp.linalg.norm(diff_e, axis=-1, keepdims=True)
        en_grad = jnp.sum(z_i * diff_e / (dist_e ** 3), axis=0)

        grads.append(nn_grad + en_grad)

    return jnp.stack(grads, axis=0)


def _finite_difference_grad(
    electrons: Electron,
    nuclei: Nucleus,
    eps: float = 1e-3,
) -> jax.Array:
    """Central finite-difference gradient of simplified_hamiltonian wrt nuclei.position."""
    num_nuclei, dim = nuclei.position.shape
    fd_grad = jnp.zeros_like(nuclei.position)
    for i in range(num_nuclei):
        for d in range(dim):
            delta = jnp.zeros_like(nuclei.position).at[i, d].set(eps)
            e_plus = simplified_hamiltonian(
                electrons, Nucleus(position=nuclei.position + delta, charge=nuclei.charge)
            )
            e_minus = simplified_hamiltonian(
                electrons, Nucleus(position=nuclei.position - delta, charge=nuclei.charge)
            )
            fd_grad = fd_grad.at[i, d].set((e_plus - e_minus) / (2.0 * eps))
    return fd_grad


def _make_batch_electrons(
    electron_positions: jax.Array,
    batch_size: int = 100,
) -> Electron:
    """Replicate or wrap batched electron positions into an Electron pytree."""
    if electron_positions.ndim == 2:
        positions = jnp.broadcast_to(
            electron_positions[None, :, :],
            (batch_size, electron_positions.shape[0], 3),
        )
    else:
        positions = electron_positions
    num_electrons = positions.shape[1]
    spins = jnp.broadcast_to(
        ((-1.0) ** jnp.arange(num_electrons))[None, :, None],
        (positions.shape[0], num_electrons, 1),
    )
    return Electron(position=positions, spin=spins)


# Geometries with moderate inter-particle distances so per-sample gradient norm < 10.0
# (below the @clip_grad_global(max_norm=10.0) threshold).
GEOMETRY_CASES = [
    pytest.param(
        "H2_2nuclei",
        jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=jnp.float32),
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        jnp.array(
            [
                [1.2, 0.4, -0.3],
                [-0.9, -0.7, 0.8],
            ],
            dtype=jnp.float32,
        ),
        id="H2_2nuclei",
    ),
    pytest.param(
        "H2O_3nuclei",
        jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.8, 1.4, 0.0],
                [-1.8, 1.4, 0.0],
            ],
            dtype=jnp.float32,
        ),
        jnp.array([8.0, 1.0, 1.0], dtype=jnp.float32),
        jnp.array(
            [
                [2.5, -2.0, 1.5],
                [-2.5, -2.0, -1.5],
                [0.5, 3.2, 1.8],
                [-0.6, 3.1, -1.9],
                [3.5, 0.5, -1.2],
                [-3.4, 0.6, 1.3],
                [1.7, -1.8, -2.6],
                [-1.6, -1.9, 2.7],
                [2.8, 2.9, 1.1],
                [-2.9, 2.8, -1.2],
            ],
            dtype=jnp.float32,
        ),
        id="H2O_3nuclei",
    ),
    pytest.param(
        "NH3_4nuclei",
        jnp.array(
            [
                [0.0, 0.0, 0.8],
                [1.9, 0.0, -0.5],
                [-0.95, 1.65, -0.5],
                [-0.95, -1.65, -0.5],
            ],
            dtype=jnp.float32,
        ),
        jnp.array([7.0, 1.0, 1.0, 1.0], dtype=jnp.float32),
        jnp.array(
            [
                [2.4, 2.1, 1.6],
                [-2.3, 2.2, 1.4],
                [0.2, -2.8, 1.7],
                [2.7, -1.5, -1.9],
                [-2.6, -1.7, -1.8],
                [0.4, 2.9, -1.6],
                [1.5, 0.5, 3.2],
                [-1.6, -0.4, 3.1],
                [0.1, 0.2, -3.3],
                [3.1, 1.2, -0.4],
            ],
            dtype=jnp.float32,
        ),
        id="NH3_4nuclei",
    ),
]


@pytest.mark.parametrize("name,nuclei_pos,nuclei_charge,electron_pos", GEOMETRY_CASES)
def test_hellmann_feynman_matches_analytical_and_finite_diff(
    name, nuclei_pos, nuclei_charge, electron_pos
):
    """Verify Hellmann-Feynman gradients against closed-form Coulomb forces and finite differences."""
    nuclei = Nucleus(position=nuclei_pos, charge=nuclei_charge)
    electron_batch = _make_batch_electrons(electron_pos, batch_size=100)
    single_electron = Electron(
        position=electron_pos,
        spin=electron_batch.spin[0],
    )

    expected_grad = _exact_coulomb_grad(electron_pos, nuclei_pos, nuclei_charge)
    # Sanity check that this geometry does not trigger gradient norm clipping (max_norm=10.0)
    assert float(jnp.linalg.norm(expected_grad)) < 10.0

    computed_grad = nuclei_energy_grad_hellmann_feynman(
        None, electron_batch, nuclei, jax.random.key(0)
    )
    fd_grad = _finite_difference_grad(single_electron, nuclei, eps=1e-3)

    chex.assert_shape(computed_grad, nuclei_pos.shape)
    chex.assert_trees_all_close(computed_grad, expected_grad, rtol=1e-4, atol=1e-4)
    chex.assert_trees_all_close(computed_grad, fd_grad, rtol=1e-2, atol=1e-3)


def test_hellmann_feynman_batch_averaging_matches_mean_analytical_grad():
    """Verify averaging over heterogeneous electron samples matches the mean analytical gradient."""
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0], [1.8, 1.4, 0.0], [-1.8, 1.4, 0.0]], dtype=jnp.float32),
        charge=jnp.array([8.0, 1.0, 1.0], dtype=jnp.float32),
    )
    num_samples = 100
    num_electrons = 10
    key = jax.random.key(42)

    # Place electrons on a shell of radius ~3.5-5.5 bohr so all per-sample norms stay well below 10.0
    raw = jax.random.normal(key, shape=(num_samples, num_electrons, 3))
    dirs = raw / jnp.linalg.norm(raw, axis=-1, keepdims=True)
    radii = jax.random.uniform(
        jax.random.key(43), shape=(num_samples, num_electrons, 1), minval=3.5, maxval=5.5
    )
    electron_positions = dirs * radii
    electron_batch = _make_batch_electrons(electron_positions)

    per_sample_exact = jax.vmap(
        lambda e_pos: _exact_coulomb_grad(e_pos, nuclei.position, nuclei.charge)
    )(electron_positions)
    assert float(jnp.max(jnp.linalg.norm(per_sample_exact, axis=(1, 2)))) < 10.0

    expected_mean_grad = jnp.mean(per_sample_exact, axis=0)
    computed_grad = nuclei_energy_grad_hellmann_feynman(
        None, electron_batch, nuclei, jax.random.key(0)
    )

    chex.assert_trees_all_close(computed_grad, expected_mean_grad, rtol=1e-4, atol=1e-4)


def test_nuclear_repulsion_not_overcounted_by_num_electrons():
    """Verify nuclear-nuclear repulsion is counted once per sample, not scaled by num_electrons."""
    nuclei = Nucleus(
        position=jnp.array(
            [[0.0, 0.0, 0.0], [2.2, 1.5, 0.0], [-2.2, 1.5, 0.0]],
            dtype=jnp.float32,
        ),
        charge=jnp.array([8.0, 1.0, 1.0], dtype=jnp.float32),
    )
    num_electrons = 10

    # Place all 10 electrons far away (~1e3 bohr) so electron-nuclear force is negligible (~1e-6)
    far_electrons = jnp.broadcast_to(
        jnp.array([[1000.0, 0.0, 0.0]], dtype=jnp.float32),
        (num_electrons, 3),
    )
    electron_batch = _make_batch_electrons(far_electrons, batch_size=100)

    # Pure nuclear-nuclear repulsion gradient (0 electrons)
    empty_electrons = jnp.zeros((0, 3), dtype=jnp.float32)
    expected_nn_grad = _exact_coulomb_grad(empty_electrons, nuclei.position, nuclei.charge)
    assert float(jnp.linalg.norm(expected_nn_grad)) < 10.0

    computed_grad = nuclei_energy_grad_hellmann_feynman(
        None, electron_batch, nuclei, jax.random.key(0)
    )

    # Must match 1x nuclear-nuclear repulsion, NOT num_electrons x nuclear-nuclear repulsion
    chex.assert_trees_all_close(computed_grad, expected_nn_grad, rtol=1e-4, atol=1e-4)
    assert not jnp.allclose(computed_grad, num_electrons * expected_nn_grad, rtol=1e-2)

    # Also verify with electrons at finite distances that electron-nuclear contribution is NOT diluted by 1/num_electrons
    near_electrons = jnp.array(
        [
            [2.5, -2.0, 1.5],
            [-2.5, -2.0, -1.5],
            [0.5, 3.2, 1.8],
            [-0.6, 3.1, -1.9],
            [3.5, 0.5, -1.2],
            [-3.4, 0.6, 1.3],
            [1.7, -1.8, -2.6],
            [-1.6, -1.9, 2.7],
            [2.8, 2.9, 1.1],
            [-2.9, 2.8, -1.2],
        ],
        dtype=jnp.float32,
    )
    near_batch = _make_batch_electrons(near_electrons, batch_size=100)
    expected_full = _exact_coulomb_grad(near_electrons, nuclei.position, nuclei.charge)
    expected_en_grad = expected_full - expected_nn_grad
    buggy_diluted_grad = expected_nn_grad + expected_en_grad / num_electrons

    computed_near = nuclei_energy_grad_hellmann_feynman(
        None, near_batch, nuclei, jax.random.key(0)
    )
    chex.assert_trees_all_close(computed_near, expected_full, rtol=1e-4, atol=1e-4)
    assert not jnp.allclose(computed_near, buggy_diluted_grad, rtol=1e-2)


@pytest.mark.parametrize("num_nuclei", [2, 3, 4])
def test_malformed_electron_shape_raises_assertion_error(num_nuclei):
    """Passing (batch, 3) electron positions (from collapsing num_electrons) must raise, not broadcast."""
    nuclei = Nucleus(
        position=jax.random.normal(jax.random.key(num_nuclei), shape=(num_nuclei, 3)),
        charge=jnp.ones((num_nuclei,), dtype=jnp.float32),
    )
    # Malformed 2D (batch, 3) shape produced by the old "n c s ... -> (n c s) ..." bug
    malformed_electrons = Electron(
        position=jax.random.normal(jax.random.key(100 + num_nuclei), shape=(100, 3)),
        spin=jnp.ones((100, 1), dtype=jnp.float32),
    )

    with pytest.raises(AssertionError):
        nuclei_energy_grad_hellmann_feynman(
            None, malformed_electrons, nuclei, jax.random.key(0)
        )

    with pytest.raises(AssertionError):
        nuclei_energy_grad_full(
            None, malformed_electrons, nuclei, jax.random.key(0)
        )


def test_simplified_hamiltonian_rejects_malformed_shapes():
    """simplified_hamiltonian must reject 1D electron positions (which previously silently broadcasted for 3 nuclei)."""
    nuclei_3 = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]),
        charge=jnp.array([8.0, 1.0, 1.0]),
    )

    # 1D electron position (3,) — previously silently broadcasted as (3, 1) - (1, 3, 3) for 3-nucleus H2O
    malformed_1d_electron = Electron(
        position=jnp.array([0.5, -0.5, 0.5]),
        spin=jnp.array([1.0]),
    )
    with pytest.raises(AssertionError):
        simplified_hamiltonian(malformed_1d_electron, nuclei_3)

    # 3D batched electron position passed directly to single-sample simplified_hamiltonian
    malformed_3d_electron = Electron(
        position=jnp.zeros((10, 2, 3)),
        spin=jnp.ones((10, 2, 1)),
    )
    with pytest.raises(AssertionError):
        simplified_hamiltonian(malformed_3d_electron, nuclei_3)

    # Malformed nuclei position shape
    valid_electron = Electron(
        position=jnp.array([[0.5, -0.5, 0.5], [-0.5, 0.5, -0.5]]),
        spin=jnp.array([[1.0], [-1.0]]),
    )
    malformed_nuclei = Nucleus(
        position=jnp.array([0.0, 0.0, 0.0]),
        charge=jnp.array([1.0]),
    )
    with pytest.raises(AssertionError):
        simplified_hamiltonian(valid_electron, malformed_nuclei)


@pytest.mark.parametrize(
    "nuclei_pos,nuclei_charge",
    [
        pytest.param(
            jnp.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=jnp.float32),
            jnp.array([1.0, 1.0], dtype=jnp.float32),
            id="H2_2nuclei",
        ),
        pytest.param(
            jnp.array([[0.0, 0.0, 0.0], [1.2, 0.8, 0.0], [-1.2, 0.8, 0.0]], dtype=jnp.float32),
            jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
            id="H3_3nuclei",
        ),
        pytest.param(
            jnp.array(
                [[0.0, 0.0, 0.8], [1.2, 0.0, -0.5], [-0.6, 1.0, -0.5], [-0.6, -1.0, -0.5]],
                dtype=jnp.float32,
            ),
            jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32),
            id="H4_4nuclei",
        ),
    ],
)
def test_sample_and_optimize_end_to_end(nuclei_pos, nuclei_charge):
    """Verify sample_and_optimize runs end-to-end without shape errors and updates nuclear positions."""
    sum_of_charges = int(jnp.sum(nuclei_charge))
    nuclei = Nucleus(position=nuclei_pos, charge=nuclei_charge)

    wave_function = _GaussianWaveFunction(alpha=0.5)
    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(nuclei.position)

    # sample_from_wavefunction uses mcmc_burnin_steps=50 and mcmc_thinning_factor=2,
    # so num_electron_samples=54 yields (54 - 50) / 2 = 2 post-burnin samples per chain,
    # and 2 * 50 = 100 total samples (divisible by reduce_vmap batch_size=100).
    new_nuclei, new_opt_state = sample_and_optimize(
        wave_function=wave_function,
        optimizer=optimizer,
        opt_state=opt_state,
        nuclei=nuclei,
        num_electron_samples=54,
        num_electron_chains=50,
        sum_of_charges=sum_of_charges,
        key=jax.random.key(0),
    )

    chex.assert_shape(new_nuclei.position, nuclei_pos.shape)
    chex.assert_trees_all_close(new_nuclei.charge, nuclei.charge)
    assert jnp.all(jnp.isfinite(new_nuclei.position))
    assert not jnp.allclose(new_nuclei.position, nuclei.position)
    assert new_opt_state is not None
