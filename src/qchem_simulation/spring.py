"""SPRING and MinSR natural-gradient optimizers for Variational Monte Carlo (VMC).

This experimental optimizer runs on a caller-supplied sample batch. Its dense
score matrix has shape (num_samples, num_parameters) and the Gram matrix has
shape (num_samples, num_samples).

Implements:
- **MinSR** (Chen & Heyl, 2024): Minimum-norm Stochastic Reconfiguration, which
  solves the dual ``N_s x N_s`` linear system instead of the primal ``P x P``
  quantum geometric tensor ``S = O^T O``:

      phi = O^T (O O^T + lambda I)^{-1} epsilon

- **SPRING** (Goldshlager, Abrahamsen & Lin, 2024): A Kaczmarz-inspired
  momentum extension of MinSR that accumulates curvature information across
  VMC iterations:

      eps_tilde = epsilon - O (mu * phi_{t-1})
      phi_t     = mu * phi_{t-1} + O^T (O O^T + lambda I)^{-1} eps_tilde
      theta_{t+1} = theta_t - eta * clip(phi_t)

Here ``O`` is the ``(N_s, P)`` matrix of centered per-sample log-amplitude
gradients ``d log|psi(x_i)| / d theta`` scaled by ``1 / sqrt(N_s)``, and
``epsilon`` is the ``(N_s,)`` vector of centered local energies
``E_L(x_i) - <E_L>`` scaled by ``1 / sqrt(N_s)``.

With this convention, ``S = O^T O`` is the Fubini-Study / QGT metric,
``f = O^T epsilon = (1/2) grad_theta E`` is the VMC force vector, and
``phi = (S + lambda I)^{-1} f = (1/2) (S + lambda I)^{-1} grad_theta E``
has the exact imaginary-time / natural-gradient scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Key

from .electronic_optimization import HamiltonianParts, WaveFunction, hamiltonian
from .utils import Electron, Nucleus

PyTree = Any


@jax.tree_util.register_dataclass
@dataclass
class SPRINGState:
    """Pytree state for the SPRING / MinSR optimizer.

    Attributes:
        phi: Previous natural-gradient direction in flattened parameter space,
            shape ``(P,)``. Represents the running estimate of ``S^{-1} f``.
        step: Number of completed optimizer steps (scalar int32 array).
    """

    phi: Array
    step: Array


def init_spring_state(params: PyTree) -> SPRINGState:
    """Initialize a ``SPRINGState`` matching the parameter pytree ``params``."""
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    return SPRINGState(
        phi=jnp.zeros_like(flat_params),
        step=jnp.array(0, dtype=jnp.int32),
    )


def compute_spring_direction(
    O: Array,
    epsilon: Array,
    prev_phi: Array,
    *,
    damping: float | Array = 1e-3,
    momentum: float | Array = 0.99,
) -> Array:
    """Compute the SPRING (or MinSR when ``momentum=0.0``) natural-gradient direction.

    Solves the ``N_s x N_s`` symmetric positive-definite linear system using
    Cholesky factorization:

        eps_tilde = epsilon - O @ (momentum * prev_phi)
        T         = O @ O.T + damping * I
        phi       = momentum * prev_phi + O.T @ T^{-1} @ eps_tilde

    Args:
        O: Centered, ``1 / sqrt(N_s)``-scaled log-amplitude Jacobian of shape ``(N_s, P)``.
        epsilon: Centered, ``1 / sqrt(N_s)``-scaled local energies of shape ``(N_s,)``.
        prev_phi: Previous direction ``phi_{t-1}`` of shape ``(P,)``.
        damping: Diagonal regularization ``lambda > 0`` added to ``O @ O.T``.
        momentum: SPRING momentum decay ``mu in [0, 1)``. Setting ``momentum=0.0``
            recovers exact MinSR.

    Returns:
        Updated natural-gradient direction ``phi_t`` of shape ``(P,)``.
    """
    num_samples = O.shape[0]
    decayed_phi = momentum * prev_phi
    eps_tilde = epsilon - O @ decayed_phi

    gram = O @ O.T
    gram = 0.5 * (gram + gram.T)
    reg_gram = gram + damping * jnp.eye(num_samples, dtype=O.dtype)

    def spectral_correction():
        # Work on O itself when squaring its condition number loses the ridge.
        left, singular_values, right_t = jnp.linalg.svd(O, full_matrices=False)
        lengths = jnp.hypot(singular_values, jnp.sqrt(damping))
        ridge_weights = (singular_values / lengths) / lengths
        return right_t.T @ (ridge_weights * (left.T @ eps_tilde))

    def cholesky_correction():
        factor = jax.scipy.linalg.cho_factor(reg_gram, lower=True)
        z = jax.scipy.linalg.cho_solve(factor, eps_tilde)
        correction = O.T @ z
        return jax.lax.cond(
            jnp.all(jnp.isfinite(correction)), lambda: correction, spectral_correction
        )

    ridge_visible = jnp.all(jnp.diag(reg_gram) > jnp.diag(gram))
    return decayed_phi + jax.lax.cond(ridge_visible, cholesky_correction, spectral_correction)


def spring_update(
    params: PyTree,
    state: SPRINGState,
    local_energies: Array,
    per_sample_log_grads: PyTree,
    *,
    learning_rate: float | Array = 1e-2,
    damping: float | Array = 1e-3,
    momentum: float | Array = 0.99,
    max_norm: float | Array | None = None,
    eps: float = 1e-12,
) -> tuple[PyTree, SPRINGState]:
    """Perform one SPRING / MinSR parameter update on a parameter pytree.

    Args:
        params: Current model parameter pytree.
        state: Current ``SPRINGState``.
        local_energies: Per-sample local energies ``E_L(x_i)`` of shape ``(N_s,)``.
        per_sample_log_grads: Pytree with the same structure as ``params`` where
            each leaf has shape ``(N_s, *param_shape)``, containing
            ``d log|psi(x_i)| / d theta`` (NOT ``2 * log|psi|``).
        learning_rate: Step size ``eta``.
        damping: Tikhonov regularization ``lambda > 0`` for the ``N_s x N_s`` solve.
        momentum: SPRING momentum ``mu``. Use ``0.0`` for plain MinSR.
        max_norm: Optional L2 trust-region norm cap applied to the parameter step
            ``delta_theta = learning_rate * phi_t``. The unclipped ``phi_t`` is
            retained in ``SPRINGState.phi`` so its scale stays matched to ``epsilon``.
        eps: Small constant for numerical safety in norm clipping.

    Returns:
        A tuple ``(new_params, new_state)``. A nonfinite proposal leaves both
        parameters and state at their previous values.
    """
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    flat_log_grads = jax.vmap(lambda g: jax.flatten_util.ravel_pytree(g)[0])(per_sample_log_grads)

    num_samples = local_energies.shape[0]
    inv_sqrt_n = jax.lax.rsqrt(jnp.asarray(num_samples, dtype=flat_log_grads.dtype))

    O = (flat_log_grads - jnp.mean(flat_log_grads, axis=0, keepdims=True)) * inv_sqrt_n
    epsilon = (local_energies - jnp.mean(local_energies)) * inv_sqrt_n

    phi = compute_spring_direction(
        O,
        epsilon,
        state.phi,
        damping=damping,
        momentum=momentum,
    )

    delta_theta = learning_rate * phi
    if max_norm is not None:
        step_norm = jnp.linalg.norm(delta_theta)
        scale = jnp.minimum(1.0, max_norm / (step_norm + eps))
        delta_theta = delta_theta * scale

    proposed_params = flat_params - delta_theta
    valid = jnp.all(jnp.isfinite(phi)) & jnp.all(jnp.isfinite(proposed_params))
    next_params = jnp.where(valid, proposed_params, flat_params)
    next_state = SPRINGState(phi=jnp.where(valid, phi, state.phi), step=state.step + valid)
    return unravel_fn(next_params), next_state


def compute_per_sample_log_grads(
    log_psi_fn: Callable[[PyTree, Any], Array],
    params: PyTree,
    samples: Any,
) -> tuple[Array, PyTree]:
    """Compute per-sample ``log|psi(x_i)|`` values and parameter gradients.

    Args:
        log_psi_fn: Pure function ``(params, single_sample) -> scalar log|psi|``.
        params: Parameter pytree.
        samples: Batch of samples with leading axis ``N_s``.

    Returns:
        Tuple ``(log_psi, per_sample_log_grads)`` where ``log_psi`` has shape
        ``(N_s,)`` and each leaf of ``per_sample_log_grads`` has shape
        ``(N_s, *param_shape)``.
    """
    return jax.vmap(jax.value_and_grad(log_psi_fn), in_axes=(None, 0))(params, samples)


def compute_local_energies_and_log_grads(
    wave_function: WaveFunction,
    samples: Electron,
    nuclei: Nucleus,
    key: Key,
) -> tuple[Array, nnx.State, HamiltonianParts, Array]:
    """Compute per-sample local energies and ``d log|psi| / d theta`` for ``WaveFunction``.

    Note that unlike ``electronic_optimization.log_prob_fn`` (which differentiates
    ``2 * log_psi``), this helper differentiates ``log_psi`` directly so the
    resulting per-sample gradients match the SPRING / MinSR convention.

    Args:
        wave_function: Flax NNX ``WaveFunction`` module.
        samples: Batch of ``Electron`` samples with leading axis ``N_s``.
        nuclei: ``Nucleus`` configuration.
        key: JAX PRNG key.

    Returns:
        Tuple ``(local_energies, per_sample_log_grads, hamiltonian_details, log_psi)``.
    """
    keys = jax.random.split(key, samples.position.shape[0])
    local_energies, hamiltonian_details = nnx.vmap(
        hamiltonian, in_axes=(None, 0, None, 0)
    )(wave_function, samples, nuclei, keys)

    graphdef, params, rest = nnx.split(wave_function, nnx.Param, ...)

    def _log_psi_fn(param_state: nnx.State, sample: Electron) -> Array:
        wf = nnx.merge(graphdef, param_state, rest)
        log_psi, _sign, _ortho_measure = wf(sample, nuclei)
        return log_psi

    log_psi, per_sample_log_grads = compute_per_sample_log_grads(
        _log_psi_fn, params, samples
    )
    return local_energies, per_sample_log_grads, hamiltonian_details, log_psi


class SpringOptimizer:
    """Flax NNX helper wrapping ``SPRINGState`` and ``spring_update``.

    Updates an ``nnx.Module``'s ``nnx.Param`` state in-place using SPRING
    (``momentum > 0``) or MinSR (``momentum = 0.0``).
    """

    def __init__(
        self,
        model: nnx.Module,
        *,
        learning_rate: float = 1e-2,
        damping: float = 1e-3,
        momentum: float = 0.99,
        max_norm: float | None = None,
    ):
        _, params, _ = nnx.split(model, nnx.Param, ...)
        self.state: SPRINGState = init_spring_state(params)
        self.learning_rate = learning_rate
        self.damping = damping
        self.momentum = momentum
        self.max_norm = max_norm

    def update(
        self,
        model: nnx.Module,
        local_energies: Array,
        per_sample_log_grads: nnx.State,
        *,
        learning_rate: float | None = None,
    ) -> SPRINGState:
        """Apply one SPRING / MinSR update in-place to ``model``."""
        _, params, _ = nnx.split(model, nnx.Param, ...)
        lr = self.learning_rate if learning_rate is None else learning_rate
        new_params, new_state = spring_update(
            params,
            self.state,
            local_energies,
            per_sample_log_grads,
            learning_rate=lr,
            damping=self.damping,
            momentum=self.momentum,
            max_norm=self.max_norm,
        )
        self.state = new_state
        nnx.update(model, new_params)
        return new_state

    def step(
        self,
        wave_function: WaveFunction,
        samples: Electron,
        nuclei: Nucleus,
        key: Key,
        *,
        learning_rate: float | None = None,
    ) -> tuple[Array, HamiltonianParts, Array]:
        """Compute local energies & log-grads for ``WaveFunction`` and update in-place.

        Returns:
            ``(avg_energy, mean_hamiltonian_details, mean_log_psi)``, matching
            ``electronic_optimization.optimize_wave_function``.
        """
        local_energies, per_sample_log_grads, hamiltonian_details, log_psi = (
            compute_local_energies_and_log_grads(wave_function, samples, nuclei, key)
        )
        self.update(
            wave_function,
            local_energies,
            per_sample_log_grads,
            learning_rate=learning_rate,
        )
        avg_energy = jnp.mean(local_energies)
        mean_hamiltonian_details = jax.tree.map(
            lambda x: jnp.mean(x, axis=0), hamiltonian_details
        )
        return avg_energy, mean_hamiltonian_details, jnp.mean(log_psi)
