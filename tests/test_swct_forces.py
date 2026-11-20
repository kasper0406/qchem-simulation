"""Unit tests for Space-Warp Coordinate Transformation (SWCT) nuclear forces."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from flax import nnx

from qchem_simulation.electronic_optimization import WaveFunction
from qchem_simulation.nuclei_optimization import nuclei_energy_grad_swct
from qchem_simulation.swct_forces import (
    swct_energy_grad,
    swct_energy_grad_from_fns,
    warp_electron_positions,
    warp_log_jacobian,
    warp_log_jacobian_grad,
    warp_weights,
)
from qchem_simulation.utils import Electron, Nucleus


# ---------------------------------------------------------------------------
# Analytical 2-center Gaussian LCAO model for H2+ (2 nuclei, 1 electron)
# ---------------------------------------------------------------------------


def _h2plus_gaussian_log_psi(
    r: jax.Array,
    nuclei_pos: jax.Array,
    alpha: float = 0.8,
) -> jax.Array:
    """Log-amplitude of psi(r; R) = exp(-alpha |r - R_A|^2) + exp(-alpha |r - R_B|^2)."""
    r0 = r[0]  # (3,)
    d_sq = jnp.sum(jnp.square(r0[None, :] - nuclei_pos), axis=-1)  # (2,)
    return jax.nn.logsumexp(-alpha * d_sq)


def _h2plus_gaussian_local_energy(
    r: jax.Array,
    nuclei_pos: jax.Array,
    charges: jax.Array,
    alpha: float = 0.8,
) -> jax.Array:
    """Exact local energy E_L(r; R) = (-1/2 nabla^2 psi / psi) + V_ne + V_nn."""
    r0 = r[0]
    diff = r0[None, :] - nuclei_pos  # (2, 3)
    d_sq = jnp.sum(jnp.square(diff), axis=-1)  # (2,)
    d = jnp.sqrt(d_sq)

    # For phi_I(r) = exp(-alpha |r - R_I|^2):
    #   nabla^2 phi_I = (4 alpha^2 d_I^2 - 6 alpha) phi_I
    # So -1/2 (nabla^2 psi / psi) = sum_I p_I * (3 alpha - 2 alpha^2 d_I^2)
    # where p_I = phi_I / (phi_A + phi_B) = softmax(-alpha * d_sq).
    p = jax.nn.softmax(-alpha * d_sq)
    kinetic = jnp.sum(p * (3.0 * alpha - 2.0 * (alpha**2) * d_sq))

    v_ne = -jnp.sum(charges / d)
    r_ab = jnp.linalg.norm(nuclei_pos[0] - nuclei_pos[1])
    v_nn = (charges[0] * charges[1]) / r_ab
    return kinetic + v_ne + v_nn


def _boys_f0(t: jax.Array) -> jax.Array:
    """Boys function F_0(t) = sqrt(pi)/(2 sqrt(t)) erf(sqrt(t)), with F_0(0) = 1."""
    safe_t = jnp.where(t > 1e-12, t, 1.0)
    sqrt_t = jnp.sqrt(safe_t)
    val = 0.5 * jnp.sqrt(jnp.pi) * jsp.erf(sqrt_t) / sqrt_t
    return jnp.where(t > 1e-12, val, 1.0 - t / 3.0)


def _h2plus_gaussian_exact_energy(
    nuclei_pos: jax.Array,
    charges: jax.Array,
    alpha: float = 0.8,
) -> jax.Array:
    """Closed-form expectation value <E(R)> for the 2-center Gaussian H2+ state."""
    r_ab = jnp.linalg.norm(nuclei_pos[0] - nuclei_pos[1])
    r_ab_sq = r_ab**2
    p = 2.0 * alpha
    s_aa = (jnp.pi / p) ** 1.5
    exp_ab = jnp.exp(-0.5 * alpha * r_ab_sq)
    s_ab = s_aa * exp_ab

    norm = 2.0 * (s_aa + s_ab)

    t_aa = 1.5 * alpha * s_aa
    t_ab = 0.5 * alpha * (3.0 - alpha * r_ab_sq) * s_ab
    kinetic = 2.0 * (t_aa + t_ab)

    pref = 2.0 * jnp.pi / p
    # <phi_A | -Z_A/r_A - Z_B/r_B | phi_A>
    v_aa = -pref * (charges[0] * 1.0 + charges[1] * _boys_f0(p * r_ab_sq))
    # <phi_B | -Z_A/r_A - Z_B/r_B | phi_B>
    v_bb = -pref * (charges[0] * _boys_f0(p * r_ab_sq) + charges[1] * 1.0)
    # 2 <phi_A | -Z_A/r_A - Z_B/r_B | phi_B> (midpoint distance to each nucleus is R_AB / 2)
    v_ab = -2.0 * pref * exp_ab * (charges[0] + charges[1]) * _boys_f0(p * 0.25 * r_ab_sq)

    v_nn = (charges[0] * charges[1]) / r_ab
    return (kinetic + v_aa + v_bb + v_ab) / norm + v_nn


def _sample_h2plus_gaussian(
    key: jax.Array,
    nuclei_pos: jax.Array,
    num_samples: int,
    alpha: float = 0.8,
) -> jax.Array:
    """Draw exact i.i.d. samples from |psi(r; R)|^2 (3-component Gaussian mixture)."""
    r_ab_sq = jnp.sum(jnp.square(nuclei_pos[0] - nuclei_pos[1]))
    exp_ab = jnp.exp(-0.5 * alpha * r_ab_sq)
    mix_weights = jnp.array([1.0, 1.0, 2.0 * exp_ab])
    mix_weights = mix_weights / jnp.sum(mix_weights)

    centers = jnp.stack(
        [nuclei_pos[0], nuclei_pos[1], 0.5 * (nuclei_pos[0] + nuclei_pos[1])],
        axis=0,
    )
    k_comp, k_noise = jax.random.split(key)
    comps = jax.random.choice(k_comp, 3, shape=(num_samples,), p=mix_weights)
    sigma = 1.0 / jnp.sqrt(4.0 * alpha)  # exp(-2 alpha |r - c|^2) -> var = 1 / (4 alpha)
    noise = jax.random.normal(k_noise, shape=(num_samples, 3)) * sigma
    samples = centers[comps] + noise
    return samples[:, None, :]  # (num_samples, 1, 3)


def _bare_hellmann_feynman_per_sample(
    electron_positions_batch: jax.Array,
    nuclei_pos: jax.Array,
    charges: jax.Array,
) -> jax.Array:
    """Bare Hellmann-Feynman gradient grad_R V(r; R) per sample."""

    def potential(r_nuc: jax.Array, r_elec: jax.Array) -> jax.Array:
        d = jnp.linalg.norm(r_elec[:, None, :] - r_nuc[None, :, :], axis=-1)
        v_ne = -jnp.sum(charges[None, :] / d)
        r_ab = jnp.linalg.norm(r_nuc[0] - r_nuc[1])
        v_nn = (charges[0] * charges[1]) / r_ab
        return v_ne + v_nn

    return jax.vmap(jax.grad(potential, argnums=0), in_axes=(None, 0))(
        nuclei_pos, electron_positions_batch
    )


# ---------------------------------------------------------------------------
# 1. Warp & Jacobian tests
# ---------------------------------------------------------------------------


def test_warp_weights_limits_and_normalization():
    r0 = jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.5, 0.0, 0.0]])
    key = jax.random.key(42)
    r_rand = jax.random.normal(key, shape=(8, 3))

    w = warp_weights(r_rand, r0)
    assert w.shape == (8, 3)
    assert jnp.all(w >= 0.0) and jnp.all(w <= 1.0)
    assert jnp.allclose(jnp.sum(w, axis=-1), 1.0, atol=1e-6)

    # As r -> R_{0,I}, w_I -> 1 and w_{J != I} -> 0
    for idx in range(r0.shape[0]):
        r_near = r0[idx : idx + 1] + jnp.array([[1e-4, -1e-4, 1e-4]])
        w_near = warp_weights(r_near, r0)[0]
        expected = jnp.zeros(3).at[idx].set(1.0)
        assert jnp.allclose(w_near, expected, atol=1e-6)

        # Exact coincidence r == R_{0,I} must be finite and equal to 1
        w_exact = warp_weights(r0[idx : idx + 1], r0)[0]
        assert jnp.all(jnp.isfinite(w_exact))
        assert jnp.allclose(w_exact, expected, atol=1e-6)

    # At R = R0, warped positions equal r and log J = 0
    assert jnp.allclose(warp_electron_positions(r_rand, r0, r0), r_rand, atol=1e-7)
    assert jnp.allclose(warp_log_jacobian(r_rand, r0, r0), 0.0, atol=1e-7)

    # Single-nucleus edge case: rigid shift, zero Jacobian gradient
    r0_single = jnp.array([[0.3, -0.5, 0.7]])
    w_single = warp_weights(r_rand, r0_single)
    assert jnp.allclose(w_single, 1.0, atol=1e-7)
    grad_log_j_single = warp_log_jacobian_grad(r_rand, r0_single)
    assert jnp.allclose(grad_log_j_single, 0.0, atol=1e-7)


def test_warp_log_jacobian_grad_matches_finite_differences():
    key = jax.random.key(7)
    k1, k2 = jax.random.split(key)
    r = jax.random.normal(k1, shape=(4, 3))
    r0 = jax.random.normal(k2, shape=(3, 3))

    # Analytic formula: sum_i grad_{r_i} w_I(r_i; R_0)
    grad_analytic = warp_log_jacobian_grad(r, r0)

    # Direct autodiff of log|det(d r_tilde / d r)| wrt R at R = R0
    grad_autodiff = jax.grad(lambda r_nuc: warp_log_jacobian(r, r_nuc, r0))(r0)
    assert jnp.allclose(grad_analytic, grad_autodiff, rtol=1e-5, atol=1e-5)

    # Central finite differences of warp_log_jacobian wrt every nuclear coordinate
    h = 1e-3
    grad_fd = jnp.zeros_like(r0)
    for i in range(r0.shape[0]):
        for d in range(3):
            e = jnp.zeros_like(r0).at[i, d].set(h)
            j_plus = warp_log_jacobian(r, r0 + e, r0)
            j_minus = warp_log_jacobian(r, r0 - e, r0)
            grad_fd = grad_fd.at[i, d].set((j_plus - j_minus) / (2.0 * h))

    assert jnp.allclose(grad_analytic, grad_fd, rtol=1e-3, atol=1e-4)

    # Translational invariance: sum_I grad_{R_I} log J == 0
    assert jnp.allclose(jnp.sum(grad_analytic, axis=0), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Unbiasedness check against exact finite differences of <E(R)>
# ---------------------------------------------------------------------------


def test_swct_mean_agrees_with_exact_finite_differences():
    alpha = 0.8
    charges = jnp.array([1.0, 1.0])
    # Asymmetric bond axis to test non-zero gradients across all 3 Cartesian axes
    r0 = jnp.array([[-0.6, -0.3, -0.5], [0.6, 0.3, 0.5]])

    # Reference gradient via central finite differences of exact <E(R)>
    h = 1e-3
    grad_exact_fd = jnp.zeros_like(r0)
    for i in range(2):
        for d in range(3):
            e = jnp.zeros_like(r0).at[i, d].set(h)
            e_plus = _h2plus_gaussian_exact_energy(r0 + e, charges, alpha=alpha)
            e_minus = _h2plus_gaussian_exact_energy(r0 - e, charges, alpha=alpha)
            grad_exact_fd = grad_exact_fd.at[i, d].set((e_plus - e_minus) / (2.0 * h))

    # Draw i.i.d. samples from |psi(r; R0)|^2
    num_samples = 40_000
    samples = _sample_h2plus_gaussian(jax.random.key(123), r0, num_samples, alpha=alpha)

    log_psi_fn = lambda r, r_nuc: _h2plus_gaussian_log_psi(r, r_nuc, alpha=alpha)
    local_e_fn = lambda r, r_nuc: _h2plus_gaussian_local_energy(r, r_nuc, charges, alpha=alpha)

    grad_swct, per_sample = swct_energy_grad_from_fns(
        log_psi_fn,
        local_e_fn,
        samples,
        r0,
        return_per_sample=True,
    )

    # Standard error of the Monte Carlo mean
    stderr = jnp.std(per_sample, axis=0) / jnp.sqrt(num_samples)

    # Compare this sampled estimate to the exact derivative within Monte Carlo error.
    z_scores = jnp.abs(grad_swct - grad_exact_fd) / (stderr + 1e-12)
    assert jnp.all(z_scores < 3.5), (
        f"SWCT grad {grad_swct} deviates from exact FD {grad_exact_fd} "
        f"(stderr={stderr}, z={z_scores})"
    )
    assert jnp.allclose(grad_swct, grad_exact_fd, rtol=0.05, atol=5e-3)

    # Exact translational invariance holds for the mean and every individual sample
    assert jnp.allclose(jnp.sum(grad_swct, axis=0), 0.0, atol=1e-6)
    net_grad_per_sample = jnp.sum(per_sample, axis=1)
    assert jnp.allclose(net_grad_per_sample, 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Variance reduction check: SWCT vs bare Hellmann-Feynman
# ---------------------------------------------------------------------------


def test_swct_variance_reduction_vs_bare_hellmann_feynman():
    alpha = 0.8
    charges = jnp.array([1.0, 1.0])
    r0 = jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])

    # Sample electrons from |psi|^2 (naturally includes electrons close to nuclei)
    num_samples = 10_000
    samples = _sample_h2plus_gaussian(jax.random.key(999), r0, num_samples, alpha=alpha)

    log_psi_fn = lambda r, r_nuc: _h2plus_gaussian_log_psi(r, r_nuc, alpha=alpha)
    local_e_fn = lambda r, r_nuc: _h2plus_gaussian_local_energy(r, r_nuc, charges, alpha=alpha)

    _, swct_per_sample = swct_energy_grad_from_fns(
        log_psi_fn,
        local_e_fn,
        samples,
        r0,
        return_per_sample=True,
    )
    hf_per_sample = _bare_hellmann_feynman_per_sample(samples, r0, charges)

    # Compare total per-sample variance across all nuclear coordinates
    swct_var = float(jnp.mean(jnp.var(swct_per_sample, axis=0)))
    hf_var = float(jnp.mean(jnp.var(hf_per_sample, axis=0)))
    reduction_natural = hf_var / swct_var
    assert reduction_natural > 20.0, (
        f"Expected >20x variance reduction on |psi|^2 samples, got {reduction_natural:.1f}x "
        f"(HF var={hf_var:.3e}, SWCT var={swct_var:.3e})"
    )

    # Now explicitly sample electrons close to nucleus 0 (r ~ 1e-3 .. 5e-2)
    k_dir, k_rad = jax.random.split(jax.random.key(2026))
    directions = jax.random.normal(k_dir, shape=(1024, 3))
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    radii = jnp.exp(jax.random.uniform(k_rad, shape=(1024, 1), minval=jnp.log(1e-3), maxval=jnp.log(5e-2)))
    near_samples = (r0[0][None, :] + directions * radii)[:, None, :]

    _, swct_near = swct_energy_grad_from_fns(
        log_psi_fn,
        local_e_fn,
        near_samples,
        r0,
        return_per_sample=True,
    )
    hf_near = _bare_hellmann_feynman_per_sample(near_samples, r0, charges)

    swct_near_var = float(jnp.mean(jnp.var(swct_near, axis=0)))
    hf_near_var = float(jnp.mean(jnp.var(hf_near, axis=0)))
    reduction_near = hf_near_var / swct_near_var
    assert reduction_near > 100.0, (
        f"Expected >100x variance reduction near nuclei, got {reduction_near:.1f}x "
        f"(HF var={hf_near_var:.3e}, SWCT var={swct_near_var:.3e})"
    )


# ---------------------------------------------------------------------------
# 4. Neural WaveFunction compatibility under nnx.jit
# ---------------------------------------------------------------------------


def test_swct_energy_grad_neural_wavefunction_jit():
    wf = WaveFunction(num_electrons=2, num_nuclei=2, rngs=nnx.Rngs(0))
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=jnp.float32),
        charge=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )

    num_samples = 6
    key = jax.random.key(0)
    pos_key, grad_key = jax.random.split(key)
    positions = jax.random.normal(pos_key, shape=(num_samples, 2, 3), dtype=jnp.float32) * 0.5
    spins = jnp.tile(jnp.array([[[1.0], [-1.0]]], dtype=jnp.float32), (num_samples, 1, 1))
    electrons = Electron(position=positions, spin=spins)

    @nnx.jit
    def compute_unbatched(wfn: WaveFunction, elec: Electron, nuc: Nucleus, k: jax.Array):
        return swct_energy_grad(wfn, elec, nuc, k)

    @nnx.jit
    def compute_batched(wfn: WaveFunction, elec: Electron, nuc: Nucleus, k: jax.Array):
        return nuclei_energy_grad_swct(wfn, elec, nuc, k, batch_size=4)

    grad_unbatched = compute_unbatched(wf, electrons, nuclei, grad_key)
    grad_batched = compute_batched(wf, electrons, nuclei, grad_key)

    assert grad_unbatched.shape == (2, 3)
    assert jnp.all(jnp.isfinite(grad_unbatched))
    assert jnp.allclose(grad_unbatched, grad_batched, rtol=1e-4, atol=1e-4)

    # Translational invariance: total force across nuclei sums to ~0
    assert jnp.allclose(jnp.sum(grad_unbatched, axis=0), 0.0, atol=1e-2)
