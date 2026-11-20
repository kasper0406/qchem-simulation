import os

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from folx import forward_laplacian

from qchem_simulation.electronic_optimization import (
    Distance,
    FlatDistanceEncoder,
    WaveFunction,
    hamiltonian,
    slater_log_sum_exp,
    slater_matrix_log_sum,
)
from qchem_simulation.main import plot_nuclei_and_electrons
from qchem_simulation.utils import Electron, Nucleus


def _assert_tree_all_finite(tree):
    leaves = jax.tree.leaves(tree)
    assert len(leaves) > 0
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        assert bool(jnp.all(jnp.isfinite(arr))), f"Non-finite values found: {arr}"


def test_flat_distance_encoder_empty_input():
    hidden_dim = 16
    encoder = FlatDistanceEncoder(hidden_dim=hidden_dim, buckets=8, rngs=nnx.Rngs(0))
    empty_distance = Distance(
        magnitude=jnp.zeros((0,), dtype=jnp.float32),
        direction=jnp.zeros((0, 3), dtype=jnp.float32),
    )

    with jax.debug_nans(True):
        out = encoder(empty_distance)

    assert out.shape == (hidden_dim,)
    assert out.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(out)))
    assert bool(jnp.all(out == 0.0))


@pytest.mark.parametrize(
    "atom_name,charge,num_electrons",
    [
        ("H", 1.0, 1),
        ("Li", 3.0, 3),
    ],
)
def test_single_nucleus_wavefunction_hamiltonian_and_grads(atom_name, charge, num_electrons):
    rngs = nnx.Rngs(42)
    key = jax.random.key(0)
    pos_key, ham_key = jax.random.split(key)

    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([charge], dtype=jnp.float32),
    )
    electron_positions = jax.random.normal(pos_key, (num_electrons, 3), dtype=jnp.float32) * 0.5
    electron_spins = jnp.reshape((-1.0) ** jnp.arange(num_electrons, dtype=jnp.float32), (num_electrons, 1))
    electrons = Electron(position=electron_positions, spin=electron_spins)

    wf = WaveFunction(num_electrons=num_electrons, num_nuclei=1, rngs=rngs)

    with jax.debug_nans(True):
        # 1. WaveFunction forward pass
        log_psi, sign, ortho = wf(electrons, nuclei)
        assert bool(jnp.isfinite(log_psi)), f"{atom_name}: log_psi is not finite"
        assert bool(jnp.isfinite(sign)), f"{atom_name}: sign is not finite"
        assert bool(jnp.isfinite(ortho)), f"{atom_name}: ortho is not finite"

        # 2. Hamiltonian (includes forward_laplacian kinetic energy)
        energy, parts = hamiltonian(wf, electrons, nuclei, ham_key)
        assert bool(jnp.isfinite(energy)), f"{atom_name}: local energy is not finite"
        _assert_tree_all_finite(parts)

        # 3. Gradients w.r.t. WaveFunction parameters
        def loss_wrt_params(model: WaveFunction):
            log_amp, _s, _o = model(electrons, nuclei)
            return log_amp

        val_p, grads_p = nnx.value_and_grad(loss_wrt_params)(wf)
        assert bool(jnp.isfinite(val_p))
        _assert_tree_all_finite(grads_p)

        # 4. Gradients w.r.t. electron positions
        def loss_wrt_positions(positions):
            e = Electron(position=positions, spin=electron_spins)
            log_amp, _s, _o = wf(e, nuclei)
            return log_amp

        val_pos, grads_pos = jax.value_and_grad(loss_wrt_positions)(electron_positions)
        assert bool(jnp.isfinite(val_pos))
        assert grads_pos.shape == electron_positions.shape
        assert bool(jnp.all(jnp.isfinite(grads_pos)))


def test_slater_log_sum_exp_standard():
    slater_logs = jnp.array([1.0, 2.0], dtype=jnp.float32)
    slater_signs = jnp.array([1.0, -1.0], dtype=jnp.float32)
    slater_coeffs = jnp.array([0.5, 0.3], dtype=jnp.float32)

    def fn(logs, coeffs):
        log_psi, _sign = slater_log_sum_exp(logs, slater_signs, coeffs)
        return log_psi

    def ref_fn(logs, coeffs):
        psi = jnp.sum(coeffs * slater_signs * jnp.exp(logs))
        return jnp.log(jnp.abs(psi))

    with jax.debug_nans(True):
        log_psi, sign_psi = slater_log_sum_exp(slater_logs, slater_signs, slater_coeffs)
        grad_logs, grad_coeffs = jax.grad(fn, argnums=(0, 1))(slater_logs, slater_coeffs)

    ref_grad_logs, ref_grad_coeffs = jax.grad(ref_fn, argnums=(0, 1))(slater_logs, slater_coeffs)

    expected_val = 0.5 * jnp.exp(1.0) - 0.3 * jnp.exp(2.0)
    assert jnp.allclose(log_psi, jnp.log(jnp.abs(expected_val)), atol=1e-5)
    assert jnp.allclose(sign_psi, jnp.sign(expected_val))
    assert jnp.allclose(grad_logs, ref_grad_logs, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(grad_coeffs, ref_grad_coeffs, rtol=1e-5, atol=1e-5)


def test_slater_log_sum_exp_singular_determinant():
    slater_logs = jnp.array([-jnp.inf, 1.0], dtype=jnp.float32)
    slater_signs = jnp.array([0.0, 1.0], dtype=jnp.float32)
    slater_coeffs = jnp.array([1.0, 2.0], dtype=jnp.float32)

    def fn(logs, coeffs):
        log_psi, _sign = slater_log_sum_exp(logs, slater_signs, coeffs)
        return log_psi

    with jax.debug_nans(True):
        log_psi, sign_psi = slater_log_sum_exp(slater_logs, slater_signs, slater_coeffs)
        grad_logs, grad_coeffs = jax.grad(fn, argnums=(0, 1))(slater_logs, slater_coeffs)

    assert jnp.allclose(log_psi, jnp.log(2.0) + 1.0, atol=1e-5)
    assert jnp.allclose(sign_psi, 1.0)
    assert bool(jnp.all(jnp.isfinite(grad_logs)))
    assert bool(jnp.all(jnp.isfinite(grad_coeffs)))


@pytest.mark.parametrize("size,zeros", [(2, 1), (4, 1), (4, 2)])
def test_slater_sum_derivatives_at_singular_orbital_matrix(size, zeros):
    identity = jnp.eye(size)

    def log_amplitude(x):
        singular = identity.at[:zeros, :zeros].set(x * jnp.eye(zeros))
        return slater_matrix_log_sum(jnp.stack((identity, singular)), jnp.ones(2))[0]

    # The sum is 1+x or 1+x**2 depending on the nullity of the changing matrix.
    expected_grad, expected_laplacian = (1.0, -1.0) if zeros == 1 else (0.0, 2.0)
    x = jnp.array(0.0)
    with jax.debug_nans(True):
        result = forward_laplacian(log_amplitude)(x)
        assert jnp.allclose(log_amplitude(x), 0.0)
        assert jnp.allclose(jax.grad(log_amplitude)(x), expected_grad)
        assert jnp.allclose(result.dense_jacobian, expected_grad)
        assert jnp.allclose(result.laplacian, expected_laplacian)


def test_slater_log_sum_exp_exact_cancellation_nodal_surface():
    slater_logs = jnp.array([1.5, 1.5], dtype=jnp.float32)
    slater_signs = jnp.array([1.0, -1.0], dtype=jnp.float32)
    slater_coeffs = jnp.array([1.0, 1.0], dtype=jnp.float32)

    def fn(logs, coeffs):
        log_psi, _sign = slater_log_sum_exp(logs, slater_signs, coeffs)
        return log_psi

    with jax.debug_nans(True):
        (log_psi, sign_psi) = slater_log_sum_exp(slater_logs, slater_signs, slater_coeffs)
        grad_logs, grad_coeffs = jax.grad(fn, argnums=(0, 1))(slater_logs, slater_coeffs)

    assert bool(jnp.isfinite(log_psi))
    assert bool(jnp.isfinite(sign_psi))
    assert bool(jnp.all(jnp.isfinite(grad_logs)))
    assert bool(jnp.all(jnp.isfinite(grad_coeffs)))


@pytest.mark.parametrize(
    "slater_coeffs",
    [
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        jnp.array([0.0, -2.5], dtype=jnp.float32),
        jnp.array([0.0, 0.0], dtype=jnp.float32),
    ],
)
def test_slater_log_sum_exp_zero_coeffs(slater_coeffs):
    slater_logs = jnp.array([0.5, -1.0], dtype=jnp.float32)
    slater_signs = jnp.array([1.0, -1.0], dtype=jnp.float32)

    def fn(logs, coeffs):
        log_psi, _sign = slater_log_sum_exp(logs, slater_signs, coeffs)
        return log_psi

    with jax.debug_nans(True):
        log_psi, sign_psi = slater_log_sum_exp(slater_logs, slater_signs, slater_coeffs)
        grad_logs, grad_coeffs = jax.grad(fn, argnums=(0, 1))(slater_logs, slater_coeffs)

    assert bool(jnp.isfinite(log_psi))
    assert bool(jnp.isfinite(sign_psi))
    assert bool(jnp.all(jnp.isfinite(grad_logs)))
    assert bool(jnp.all(jnp.isfinite(grad_coeffs)))


def test_plot_nuclei_and_electrons_creates_output_dir(tmp_path):
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charge=jnp.array([1.0], dtype=jnp.float32),
    )
    wf = WaveFunction(num_electrons=1, num_nuclei=1, rngs=nnx.Rngs(0))
    out_dir = tmp_path / "nested" / "electron_samples"
    assert not out_dir.exists()

    plot_nuclei_and_electrons(
        step=3,
        wave_function=wf,
        num_electron_chains=2,
        num_electron_samples=4,
        nuclei=nuclei,
        key=jax.random.key(0),
        output_dir=str(out_dir),
    )

    output_file = out_dir / "step_3.html"
    assert out_dir.is_dir()
    assert output_file.is_file()
    assert os.path.getsize(output_file) > 0
