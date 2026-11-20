"""Space-Warp Coordinate Transformation (SWCT) nuclear force estimator.

Implements the exact, finite-variance SWCT nuclear energy gradient estimator
(Umrigar 1989; Filippi & Umrigar 2000; Sorella & Capriotti 2010) for neural
Variational Monte Carlo (VMC).

For reference nuclear coordinates ``R_0`` and electron coordinates ``r_i``, the
warped electron coordinates are defined as:

    r_tilde_i(R) = r_i + sum_I w_I(r_i; R_0) * (R_I - R_{0,I})

with smooth quartic warp weights:

    w_I(r_i; R_0) = f(|r_i - R_{0,I}|) / sum_J f(|r_i - R_{0,J}|),
    f(d) = (d^2 + eps)^(-power / 2)   (default power=4 -> (d^2 + eps)^(-2)).

Defining the warped log-weight:

    log W(R) = 2 * log|psi(r_tilde(R); R)| + log J(R),
    log J(R) = sum_i log|det(d r_tilde_i(R) / d r_i)|,

and warped local energy ``E_L_tilde(R) = E_L(r_tilde(R); R)``, differentiating
``<E(R)>`` at ``R = R_0`` yields the exact, unbiased SWCT energy gradient:

    grad_R <E> = E[grad_R E_L_tilde] + E[(E_L_tilde - <E_L>) * grad_R log W].

Because ``w_I(r_i) -> 1`` as ``r_i -> R_{0,I}``, ``r_tilde_i(R) - R_I`` becomes
constant with respect to ``R_I`` near nucleus ``I``, cancelling the ``1 / r^2``
Hellmann-Feynman singularity and yielding a finite-variance estimator.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Key

from .electronic_optimization import WaveFunction, hamiltonian
from .utils import Electron, Nucleus


def warp_weights(
    electron_positions: Array,
    reference_nuclei_positions: Array,
    *,
    eps: float = 1e-12,
    power: float = 4.0,
) -> Array:
    """Compute smooth normalized space-warp weights ``w_I(r_i; R_0)``.

    Args:
        electron_positions: Electron coordinates of shape ``(..., num_electrons, 3)``
            or ``(3,)`` for a single electron.
        reference_nuclei_positions: Reference nuclear coordinates ``R_0`` of
            shape ``(num_nuclei, 3)``.
        eps: Small smoothing constant preventing division by zero at ``d = 0``.
        power: Decay exponent ``kappa`` in ``f(d) = (d^2 + eps)^(-kappa / 2)``.
            Defaults to ``4.0`` (``(d^2 + eps)^(-2)``).

    Returns:
        Normalized warp weights of shape ``(..., num_electrons, num_nuclei)``
        (or ``(num_nuclei,)`` for a single electron), summing to 1 along the
        last axis.
    """
    diff = electron_positions[..., None, :] - reference_nuclei_positions
    dist_sq = jnp.sum(jnp.square(diff), axis=-1)
    log_f = -0.5 * power * jnp.log(dist_sq + eps)
    return jax.nn.softmax(log_f, axis=-1)


def warp_electron_positions(
    electron_positions: Array,
    nuclei_positions: Array,
    reference_nuclei_positions: Array,
    *,
    eps: float = 1e-12,
    power: float = 4.0,
) -> Array:
    """Map electron coordinates ``r`` to warped coordinates ``r_tilde(R)``.

    Args:
        electron_positions: Electron coordinates of shape ``(num_electrons, 3)``.
        nuclei_positions: Perturbed nuclear coordinates ``R`` of shape
            ``(num_nuclei, 3)``.
        reference_nuclei_positions: Fixed reference nuclear coordinates ``R_0``
            of shape ``(num_nuclei, 3)``.
        eps: Smoothing constant for warp weights.
        power: Decay exponent for warp weights.

    Returns:
        Warped electron coordinates ``r_tilde(R)`` of shape ``(num_electrons, 3)``.
    """
    weights = warp_weights(
        electron_positions,
        reference_nuclei_positions,
        eps=eps,
        power=power,
    )
    delta_r = nuclei_positions - reference_nuclei_positions
    return electron_positions + jnp.einsum("ij,jd->id", weights, delta_r)


def warp_log_jacobian(
    electron_positions: Array,
    nuclei_positions: Array,
    reference_nuclei_positions: Array,
    *,
    eps: float = 1e-12,
    power: float = 4.0,
) -> Array:
    """Compute the log-Jacobian ``log J(R) = sum_i log|det(d r_tilde_i / d r_i)|``.

    Since each electron is warped independently, the ``3N x 3N`` Jacobian is
    block-diagonal with ``N`` blocks of size ``3 x 3``.

    Args:
        electron_positions: Electron coordinates of shape ``(num_electrons, 3)``.
        nuclei_positions: Nuclear coordinates ``R`` of shape ``(num_nuclei, 3)``.
        reference_nuclei_positions: Fixed reference nuclear coordinates ``R_0``
            of shape ``(num_nuclei, 3)``.
        eps: Smoothing constant for warp weights.
        power: Decay exponent for warp weights.

    Returns:
        Scalar log-Jacobian ``log J(R)``.
    """
    delta_r = nuclei_positions - reference_nuclei_positions

    def warp_single_electron(r_i: Array) -> Array:
        w_i = warp_weights(r_i, reference_nuclei_positions, eps=eps, power=power)
        return r_i + jnp.einsum("j,jd->d", w_i, delta_r)

    jac_blocks = jax.vmap(jax.jacfwd(warp_single_electron))(electron_positions)
    _, logabsdet = jnp.linalg.slogdet(jac_blocks)
    return jnp.sum(logabsdet)


def warp_log_jacobian_grad(
    electron_positions: Array,
    reference_nuclei_positions: Array,
    *,
    eps: float = 1e-12,
    power: float = 4.0,
) -> Array:
    """Compute ``grad_R log J(R)`` evaluated at ``R = R_0``.

    At ``R = R_0``, ``d r_tilde_i / d r_i = I_3``, so by Jacobi's formula:

        (d log J / d R_{I,a})|_{R=R_0} = sum_i d w_I(r_i; R_0) / d r_{i,a}.

    Args:
        electron_positions: Electron coordinates of shape ``(num_electrons, 3)``.
        reference_nuclei_positions: Reference nuclear coordinates ``R_0`` of
            shape ``(num_nuclei, 3)``.
        eps: Smoothing constant for warp weights.
        power: Decay exponent for warp weights.

    Returns:
        Gradient ``grad_R log J(R)|_{R=R_0}`` of shape ``(num_nuclei, 3)``.
    """

    def weights_single_electron(r_i: Array) -> Array:
        return warp_weights(r_i, reference_nuclei_positions, eps=eps, power=power)

    # Shape: (num_electrons, num_nuclei, 3)
    dw_dr = jax.vmap(jax.jacrev(weights_single_electron))(electron_positions)
    return jnp.sum(dw_dr, axis=0)


def swct_per_sample_terms(
    log_psi_fn: Callable[[Array, Array], Array],
    local_energy_fn: Callable[[Array, Array], Array],
    electron_positions: Array,
    nuclei_positions: Array,
    *,
    eps: float = 1e-12,
    power: float = 4.0,
) -> tuple[Array, Array, Array]:
    """Compute single-sample SWCT quantities ``(E_L, grad_R E_L_tilde, grad_R log W)``.

    Args:
        log_psi_fn: Callable ``(electron_positions, nuclei_positions) -> scalar``
            returning ``log|psi(r; R)|``.
        local_energy_fn: Callable ``(electron_positions, nuclei_positions) -> scalar``
            returning the local energy ``E_L(r; R) = (H psi)(r; R) / psi(r; R)``.
        electron_positions: Single electron configuration of shape ``(num_electrons, 3)``.
        nuclei_positions: Nuclear coordinates ``R_0`` of shape ``(num_nuclei, 3)``.
        eps: Smoothing constant for warp weights.
        power: Decay exponent for warp weights.

    Returns:
        Tuple ``(local_energy, grad_local_energy, grad_log_weight)`` where
        ``local_energy`` is scalar and both gradients have shape ``(num_nuclei, 3)``.
    """
    ref_nuclei = jax.lax.stop_gradient(nuclei_positions)

    def warped_local_energy(r_nuc: Array) -> Array:
        r_tilde = warp_electron_positions(
            electron_positions, r_nuc, ref_nuclei, eps=eps, power=power
        )
        return local_energy_fn(r_tilde, r_nuc)

    def warped_log_psi(r_nuc: Array) -> Array:
        r_tilde = warp_electron_positions(
            electron_positions, r_nuc, ref_nuclei, eps=eps, power=power
        )
        return log_psi_fn(r_tilde, r_nuc)

    local_e, grad_local_e = jax.value_and_grad(warped_local_energy)(nuclei_positions)
    grad_log_psi = jax.grad(warped_log_psi)(nuclei_positions)
    grad_log_j = warp_log_jacobian_grad(
        electron_positions, ref_nuclei, eps=eps, power=power
    )
    grad_log_w = 2.0 * grad_log_psi + grad_log_j
    return local_e, grad_local_e, grad_log_w


def swct_energy_grad_from_fns(
    log_psi_fn: Callable[[Array, Array], Array],
    local_energy_fn: Callable[[Array, Array], Array],
    electron_positions_batch: Array,
    nuclei_positions: Array,
    *,
    eps: float = 1e-12,
    power: float = 4.0,
    return_per_sample: bool = False,
) -> Array | tuple[Array, Array]:
    """Compute the SWCT nuclear energy gradient from arbitrary pure JAX callables.

    Args:
        log_psi_fn: Callable ``(electron_positions, nuclei_positions) -> scalar``
            returning ``log|psi(r; R)|``.
        local_energy_fn: Callable ``(electron_positions, nuclei_positions) -> scalar``
            returning the local energy ``E_L(r; R)``.
        electron_positions_batch: Batch of electron configurations of shape
            ``(num_samples, num_electrons, 3)`` sampled from ``|psi(.; R_0)|^2``.
        nuclei_positions: Nuclear coordinates ``R_0`` of shape ``(num_nuclei, 3)``.
        eps: Smoothing constant for warp weights.
        power: Decay exponent for warp weights.
        return_per_sample: If ``True``, also return per-sample SWCT gradient
            estimates of shape ``(num_samples, num_nuclei, 3)``.

    Returns:
        Mean SWCT energy gradient ``grad_R <E>`` of shape ``(num_nuclei, 3)``,
        or ``(mean_grad, per_sample_grads)`` when ``return_per_sample=True``.
    """

    def single_sample(r_elec: Array) -> tuple[Array, Array, Array]:
        return swct_per_sample_terms(
            log_psi_fn,
            local_energy_fn,
            r_elec,
            nuclei_positions,
            eps=eps,
            power=power,
        )

    local_energies, grad_local_energies, grad_log_weights = jax.vmap(single_sample)(
        electron_positions_batch
    )
    mean_energy = jnp.mean(local_energies)
    centered_energies = local_energies - mean_energy
    per_sample_grads = (
        grad_local_energies + centered_energies[:, None, None] * grad_log_weights
    )
    mean_grad = jnp.mean(per_sample_grads, axis=0)
    if return_per_sample:
        return mean_grad, per_sample_grads
    return mean_grad


def swct_energy_grad(
    wave_function: WaveFunction,
    electron_samples: Electron,
    nuclei: Nucleus,
    key: Key,
    *,
    eps: float = 1e-12,
    power: float = 4.0,
    batch_size: int | None = None,
) -> Array:
    """Compute the unbiased SWCT nuclear energy gradient for a neural ``WaveFunction``.

    Compatible with ``nnx.jit`` and matches the positional signature of
    ``nuclei_energy_grad_full`` and ``nuclei_energy_grad_hellmann_feynman``.

    Args:
        wave_function: Neural ``WaveFunction`` module.
        electron_samples: Batch of ``Electron`` samples with ``position`` of shape
            ``(num_samples, num_electrons, 3)`` and ``spin`` of shape
            ``(num_samples, num_electrons, 1)``.
        nuclei: ``Nucleus`` dataclass with ``position`` of shape ``(num_nuclei, 3)``
            and ``charge`` of shape ``(num_nuclei,)``.
        key: JAX PRNG key.
        eps: Smoothing constant for warp weights.
        power: Decay exponent for warp weights.
        batch_size: Optional mini-batch size for memory-bounded accumulation
            via ``nnx.scan``. If ``None`` or ``>= num_samples``, evaluates all
            samples in a single vectorized pass.

    Returns:
        SWCT nuclear energy gradient ``grad_R <E>`` of shape ``(num_nuclei, 3)``.
    """
    ref_nuclei_pos = jax.lax.stop_gradient(nuclei.position)
    nuclei_charges = nuclei.charge

    def per_sample_terms(
        wf: WaveFunction,
        electron_sample: Electron,
        sample_key: Key,
    ) -> tuple[Array, Array, Array]:
        r_elec = electron_sample.position
        spins = electron_sample.spin

        def warped_local_energy(r_nuc: Array, wf_inner: WaveFunction) -> Array:
            r_tilde = warp_electron_positions(
                r_elec, r_nuc, ref_nuclei_pos, eps=eps, power=power
            )
            energy, _details = hamiltonian(
                wf_inner,
                Electron(position=r_tilde, spin=spins),
                Nucleus(position=r_nuc, charge=nuclei_charges),
                sample_key,
            )
            return energy

        def warped_log_psi(r_nuc: Array, wf_inner: WaveFunction) -> Array:
            r_tilde = warp_electron_positions(
                r_elec, r_nuc, ref_nuclei_pos, eps=eps, power=power
            )
            log_psi, _sign, _ortho = wf_inner(
                Electron(position=r_tilde, spin=spins),
                Nucleus(position=r_nuc, charge=nuclei_charges),
            )
            return log_psi

        local_e, grad_local_e = nnx.value_and_grad(warped_local_energy, argnums=0)(
            nuclei.position, wf
        )
        grad_log_psi = nnx.grad(warped_log_psi, argnums=0)(nuclei.position, wf)
        grad_log_j = warp_log_jacobian_grad(
            r_elec, ref_nuclei_pos, eps=eps, power=power
        )
        grad_log_w = 2.0 * grad_log_psi + grad_log_j
        return local_e, grad_local_e, grad_log_w

    num_samples = electron_samples.position.shape[0]
    sample_keys = jax.random.split(key, num_samples)

    vmap_terms = nnx.vmap(per_sample_terms, in_axes=(None, 0, 0), out_axes=0)

    if batch_size is None or batch_size >= num_samples:
        local_energies, grad_local_energies, grad_log_weights = vmap_terms(
            wave_function, electron_samples, sample_keys
        )
        mean_energy = jnp.mean(local_energies)
        centered_energies = local_energies - mean_energy
        return jnp.mean(
            grad_local_energies + centered_energies[:, None, None] * grad_log_weights,
            axis=0,
        )

    # Batched accumulation using linearity of expectation:
    #   E[(E_L - <E_L>) * grad_log_W] = E[E_L * grad_log_W] - <E_L> * E[grad_log_W]
    num_full_batches = num_samples // batch_size
    num_full_samples = num_full_batches * batch_size
    remainder = num_samples - num_full_samples

    full_samples = jax.tree.map(
        lambda x: x[:num_full_samples].reshape((num_full_batches, batch_size) + x.shape[1:]),
        electron_samples,
    )
    full_keys = sample_keys[:num_full_samples].reshape(
        (num_full_batches, batch_size) + sample_keys.shape[1:]
    )

    zero_grad = jnp.zeros_like(nuclei.position)
    zero_scalar = jnp.zeros((), dtype=nuclei.position.dtype)

    @nnx.scan(in_axes=(None, nnx.Carry, 0, 0), out_axes=nnx.Carry)
    def accumulate_batch(wf: WaveFunction, carry, batch_electrons: Electron, batch_keys: Key):
        e_sum, grad_e_sum, grad_w_sum, e_grad_w_sum = carry
        local_e, grad_e, grad_w = vmap_terms(wf, batch_electrons, batch_keys)
        return (
            e_sum + jnp.sum(local_e, axis=0),
            grad_e_sum + jnp.sum(grad_e, axis=0),
            grad_w_sum + jnp.sum(grad_w, axis=0),
            e_grad_w_sum + jnp.sum(local_e[:, None, None] * grad_w, axis=0),
        )

    e_sum, grad_e_sum, grad_w_sum, e_grad_w_sum = accumulate_batch(
        wave_function,
        (zero_scalar, zero_grad, zero_grad, zero_grad),
        full_samples,
        full_keys,
    )

    if remainder > 0:
        rem_samples = jax.tree.map(lambda x: x[num_full_samples:], electron_samples)
        rem_keys = sample_keys[num_full_samples:]
        local_e, grad_e, grad_w = vmap_terms(wave_function, rem_samples, rem_keys)
        e_sum = e_sum + jnp.sum(local_e, axis=0)
        grad_e_sum = grad_e_sum + jnp.sum(grad_e, axis=0)
        grad_w_sum = grad_w_sum + jnp.sum(grad_w, axis=0)
        e_grad_w_sum = e_grad_w_sum + jnp.sum(local_e[:, None, None] * grad_w, axis=0)

    mean_e = e_sum / num_samples
    mean_grad_e = grad_e_sum / num_samples
    mean_grad_w = grad_w_sum / num_samples
    mean_e_grad_w = e_grad_w_sum / num_samples
    return mean_grad_e + (mean_e_grad_w - mean_e * mean_grad_w)
