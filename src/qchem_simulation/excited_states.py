"""Natural Excited States Variational Monte Carlo (NES-VMC).

Implements the Natural Excited States (NES-VMC) framework from:
    Pfau, D., Axelrod, S., Sutterud, H., von Glehn, I., & Spencer, J. S. (2024).
    "Accurate computation of quantum excited states with neural networks."
    Science, 385(6711), eadn0137.

Core idea
---------
Given K unnormalized single-state wavefunctions psi_1, ..., psi_K and K
configurations X = (x^(1), ..., x^(K)), form the K x K generalized Slater
matrix M_{ij} = psi_i(x^(j)) and the generalized total wavefunction:

    Psi(X) = det(M(X))

Minimizing the total energy E[sum_j H_j] of Psi(X) via standard VMC
automatically drives span{psi_1, ..., psi_K} to the lowest-energy K-dimensional
eigenspace of H without any orthogonality penalty hyperparameters. Individual
eigenenergies E_0 <= E_1 <= ... <= E_{K-1} and excitation gaps are recovered by
subspace diagonalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key


@dataclass
class ExcitedStateResult:
    """Result of subspace diagonalization for K excited states.

    Attributes:
        energies: Sorted eigenenergies E_0 <= E_1 <= ... <= E_{K-1}, shape (K,).
        excitation_gaps: Excitation energies E_k - E_0 for k = 1..K-1, shape (K-1,).
        eigenvectors: Generalized eigenvectors C (columns) satisfying H C = S C diag(E),
            shape (K, K), or None if not computed.
        overlap_matrix: Estimated K x K overlap matrix S_{ab}, or None.
        hamiltonian_matrix: Estimated K x K Hamiltonian matrix H_{ab}, or None.
    """

    energies: Array
    excitation_gaps: Array
    eigenvectors: Array | None = None
    overlap_matrix: Array | None = None
    hamiltonian_matrix: Array | None = None


def build_shifted_matrix(
    log_abs_matrix: Array,
    sign_matrix: Array,
) -> tuple[Array, Array]:
    """Build a numerically stable column-shifted K x K wavefunction matrix.

    For each configuration column j, shifts log|M_{ij}| by
    s_j = stop_gradient(max_i log|M_{ij}|) so that |M_tilde_{ij}| <= 1 with
    at least one unit-magnitude entry per column (unless the column is identically
    zero).

    Args:
        log_abs_matrix: Array of shape (K, K) containing log|psi_i(x^(j))|.
        sign_matrix: Array of shape (K, K) containing sign(psi_i(x^(j))).

    Returns:
        (M_tilde, col_shifts) where M_tilde has shape (K, K) and col_shifts has
        shape (K,). Satisfies det(M) = det(M_tilde) * exp(sum(col_shifts)).
    """
    raw_shifts = jnp.max(log_abs_matrix, axis=0)
    col_shifts = jax.lax.stop_gradient(
        jnp.where(jnp.isfinite(raw_shifts), raw_shifts, 0.0)
    )
    M_tilde = sign_matrix * jnp.exp(log_abs_matrix - col_shifts[None, :])
    return M_tilde, col_shifts


class MultiStateAnsatz:
    """Generalized K-state determinant ansatz Psi(X) = det(M(X)) for NES-VMC.

    Wraps a single-state evaluator ``single_state_fn(params, x) -> (log_abs, sign)``
    that returns arrays of shape ``(K,)`` for a single configuration ``x``.
    """

    def __init__(
        self,
        num_states: int,
        single_state_fn: Callable[[Any, Array], tuple[Array, Array]],
    ) -> None:
        if num_states < 1:
            raise ValueError(f"num_states must be >= 1, got {num_states}")
        self.num_states = num_states
        self.single_state_fn = single_state_fn

    @classmethod
    def from_state_fns(
        cls,
        state_fns: Sequence[Callable[[Any, Array], tuple[Array, Array]]],
    ) -> "MultiStateAnsatz":
        """Construct a MultiStateAnsatz from K individual state callables.

        Each callable ``fn(params, x)`` must return scalar ``(log_abs_psi, sign)``.
        """
        fns = tuple(state_fns)
        num_states = len(fns)

        def stacked_fn(params: Any, x: Array) -> tuple[Array, Array]:
            outs = [fn(params, x) for fn in fns]
            log_abs = jnp.stack([jnp.asarray(o[0]) for o in outs], axis=0)
            signs = jnp.stack([jnp.asarray(o[1]) for o in outs], axis=0)
            return log_abs, signs

        return cls(num_states=num_states, single_state_fn=stacked_fn)

    def eval_states(self, params: Any, x: Array) -> tuple[Array, Array]:
        """Evaluate all K single-state wavefunctions at a single configuration x.

        Returns:
            (log_abs_psi, sign) each of shape (K,).
        """
        return self.single_state_fn(params, x)

    def eval_log_matrix(self, params: Any, X: Array) -> tuple[Array, Array]:
        """Evaluate the K x K matrix M_{ij} = psi_i(x^(j)) in log-sign form.

        Args:
            params: Model parameters pytree.
            X: K configurations with leading axis K, shape (K, ...).

        Returns:
            (log_abs_M, sign_M) each of shape (K, K) where axis 0 indexes state i
            and axis 1 indexes configuration j.
        """
        # vmap over configurations j -> shape (K_configs, K_states), then transpose to (i, j)
        log_abs_T, sign_T = jax.vmap(lambda x: self.eval_states(params, x))(X)
        return log_abs_T.T, sign_T.T

    def __call__(self, params: Any, X: Array) -> tuple[Array, Array]:
        """Evaluate generalized state Psi(X) = det(M(X)) stably in log-space.

        Args:
            params: Model parameters pytree.
            X: K configurations with leading axis K, shape (K, ...).

        Returns:
            (log_abs_Psi, sign_Psi) scalar arrays.
        """
        log_abs_M, sign_M = self.eval_log_matrix(params, X)
        M_tilde, col_shifts = build_shifted_matrix(log_abs_M, sign_M)
        sign_det, log_abs_det_tilde = jnp.linalg.slogdet(M_tilde)
        log_abs_psi = log_abs_det_tilde + jnp.sum(col_shifts)
        return log_abs_psi, sign_det

    def log_prob(self, params: Any, X: Array) -> Array:
        """Unnormalized log-density log|Psi(X)|^2 for MCMC sampling."""
        log_abs_psi, _ = self(params, X)
        return 2.0 * log_abs_psi

    def mixture_log_prob(self, params: Any, x: Array) -> Array:
        """Unnormalized log-density log(sum_k |psi_k(x)|^2) for single-config sampling."""
        log_abs, _ = self.eval_states(params, x)
        return jax.scipy.special.logsumexp(2.0 * log_abs, axis=0)


def make_schrodinger_local_energy_fn(
    ansatz: MultiStateAnsatz,
    potential_fn: Callable[[Array], Array],
) -> Callable[[Any, Array], Array]:
    """Create a local energy evaluator for H = -1/2 * nabla^2 + V(x) in atomic units.

    Computes E_{L, i}(x) = -1/2 * (nabla^2 log|psi_i(x)| + ||nabla log|psi_i(x)|||^2) + V(x)
    for all K states simultaneously via forward-over-reverse autodiff.

    Args:
        ansatz: MultiStateAnsatz evaluating K states.
        potential_fn: Function mapping a single configuration x to scalar potential V(x).

    Returns:
        A callable ``local_energy_fn(params, x) -> Array`` of shape ``(K,)``.
    """

    def local_energy_fn(params: Any, x: Array) -> Array:
        orig_shape = x.shape
        x_flat = x.reshape(-1)

        def log_abs_flat(xf: Array) -> Array:
            log_abs, _ = ansatz.eval_states(params, xf.reshape(orig_shape))
            return log_abs

        grads = jax.jacrev(log_abs_flat)(x_flat)  # (K, D)
        hess = jax.jacfwd(jax.jacrev(log_abs_flat))(x_flat)  # (K, D, D)
        laplacians = jnp.einsum("kdd->k", hess)  # (K,)
        kinetic = -0.5 * (laplacians + jnp.sum(grads * grads, axis=-1))
        potential = potential_fn(x)
        return kinetic + potential

    return local_energy_fn


def nes_local_energy_matrix(
    ansatz: MultiStateAnsatz,
    params: Any,
    X: Array,
    local_energy_fn: Callable[[Any, Array], Array],
) -> Array:
    """Compute the K x K local energy matrix L(X) = (H M(X)) M(X)^{-1}.

    Column shifts cancel identically:
        (H M_tilde) M_tilde^{-1} = (H M) D D^{-1} M^{-1} = (H M) M^{-1}.

    The trace of L(X) is the total NES local energy E_L(X), and its expectation
    under X ~ |Psi(X)|^2 equals H S^{-1}, whose eigenvalues are the subspace
    eigenenergies.

    Args:
        ansatz: MultiStateAnsatz.
        params: Model parameters pytree.
        X: K configurations of shape (K, ...).
        local_energy_fn: Callable (params, x) -> (K,) single-state local energies.

    Returns:
        K x K matrix L(X).
    """
    log_abs_M, sign_M = ansatz.eval_log_matrix(params, X)
    M_tilde, _ = build_shifted_matrix(log_abs_M, sign_M)

    # Evaluate single-state local energies at each configuration j: (K_configs, K_states) -> (K_states, K_configs)
    E_L = jax.vmap(lambda x: local_energy_fn(params, x))(X).T
    H_M_tilde = jnp.where(M_tilde == 0.0, 0.0, E_L * M_tilde)

    # Solve M_tilde^T L^T = H_M_tilde^T  =>  L = H_M_tilde @ M_tilde^{-1}
    return jnp.linalg.solve(M_tilde.T, H_M_tilde.T).T


def nes_local_energy(
    ansatz: MultiStateAnsatz,
    params: Any,
    X: Array,
    local_energy_fn: Callable[[Any, Array], Array],
) -> Array:
    """Compute the total local energy of the K-state determinant ansatz.

    By Jacobi's formula / multilinearity of the determinant:
        E_L(X) = (sum_j H_j Psi(X)) / Psi(X)
               = tr(M^{-1} (H M))
               = sum_{i, j} (M^{-1})_{ji} M_{ij} E_{L, ij}

    This expression is invariant to column rescalings of M (so log-space column
    shifting cancels out exactly) and requires no orthogonality penalties.

    Args:
        ansatz: MultiStateAnsatz.
        params: Model parameters pytree.
        X: K configurations of shape (K, ...).
        local_energy_fn: Callable (params, x) -> (K,) single-state local energies.

    Returns:
        Scalar total local energy E_L(X).
    """
    log_abs_M, sign_M = ansatz.eval_log_matrix(params, X)
    M_tilde, _ = build_shifted_matrix(log_abs_M, sign_M)

    E_L = jax.vmap(lambda x: local_energy_fn(params, x))(X).T  # (K, K)
    H_M_tilde = jnp.where(M_tilde == 0.0, 0.0, E_L * M_tilde)

    # tr(M_tilde^{-1} H_M_tilde) via linear solve for numerical stability
    return jnp.trace(jnp.linalg.solve(M_tilde, H_M_tilde))


def nes_vmc_step(
    ansatz: MultiStateAnsatz,
    params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    X_batch: Array,
    local_energy_fn: Callable[[Any, Array], Array],
    *,
    clip_factor: float | None = 5.0,
) -> tuple[Any, Any, dict[str, Array]]:
    """Perform one NES-VMC optimization step using the score-function gradient.

    Computes the unbiased VMC gradient:
        g = E_{X ~ |Psi(X)|^2}[ (E_L(X) - <E_L>) * grad_theta log|Psi(X)|^2 ]

    and applies an Optax optimizer update.

    Args:
        ansatz: MultiStateAnsatz.
        params: Model parameters pytree.
        opt_state: Current Optax optimizer state.
        optimizer: Optax GradientTransformation.
        X_batch: Batch of K-configuration samples of shape (B, K, ...).
        local_energy_fn: Callable (params, x) -> (K,) single-state local energies.
        clip_factor: Optional MAD clipping multiplier for local energies to
            suppress variance spikes near nodes. Set to None to disable clipping.

    Returns:
        (new_params, new_opt_state, metrics) where metrics contains
        'total_energy', 'energy_std', and 'grad_norm'.
    """
    local_energies = jax.vmap(
        lambda X: nes_local_energy(ansatz, params, X, local_energy_fn)
    )(X_batch)

    mean_energy = jnp.mean(local_energies)
    energy_std = jnp.std(local_energies)

    if clip_factor is not None:
        median = jnp.median(local_energies)
        tv = jnp.mean(jnp.abs(local_energies - median))
        clipped_energies = jnp.clip(
            local_energies,
            median - clip_factor * tv,
            median + clip_factor * tv,
        )
    else:
        clipped_energies = local_energies

    centered = jax.lax.stop_gradient(clipped_energies - jnp.mean(clipped_energies))

    def surrogate_loss(p: Any) -> Array:
        log_probs = jax.vmap(lambda X: ansatz.log_prob(p, X))(X_batch)
        return jnp.mean(centered * log_probs)

    grads = jax.grad(surrogate_loss)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    metrics = {
        "total_energy": mean_energy,
        "energy_std": energy_std,
        "grad_norm": optax.global_norm(grads),
    }
    return new_params, new_opt_state, metrics


def solve_generalized_eigenvalue(
    H: Array,
    S: Array,
    *,
    reg: float = 1e-8,
) -> tuple[Array, Array]:
    """Solve the symmetric generalized eigenvalue problem H c = E S c.

    Uses Löwdin / spectral whitening of S with eigenvalue floor ``reg`` for
    numerical stability.

    Args:
        H: K x K Hamiltonian matrix (symmetrized internally).
        S: K x K positive semi-definite overlap matrix (symmetrized internally).
        reg: Minimum eigenvalue floor for S^{-1/2}.

    Returns:
        (energies, eigenvectors) where energies are sorted ascending of shape (K,)
        and eigenvectors has shape (K, K) with columns satisfying C^T S C = I.
    """
    H_sym = 0.5 * (H + H.T)
    S_sym = 0.5 * (S + S.T)

    s_vals, s_vecs = jnp.linalg.eigh(S_sym)
    s_vals_safe = jnp.maximum(s_vals, reg)
    whitener = s_vecs * (1.0 / jnp.sqrt(s_vals_safe))[None, :]

    H_orth = whitener.T @ H_sym @ whitener
    H_orth = 0.5 * (H_orth + H_orth.T)

    energies, eigvecs_orth = jnp.linalg.eigh(H_orth)
    eigenvectors = whitener @ eigvecs_orth
    return energies, eigenvectors


def compute_excited_state_energies(
    ansatz: MultiStateAnsatz,
    params: Any,
    samples: Array,
    local_energy_fn: Callable[[Any, Array], Array],
    *,
    log_proposal_prob: Array | Callable[[Array], Array] | None = None,
    reg: float = 1e-8,
) -> ExcitedStateResult:
    """Estimate individual excited-state energies via subspace diagonalization.

    Estimates the K x K overlap matrix S_{ab} and Hamiltonian matrix H_{ab} from
    single-configuration samples x ~ p(x) using importance weighting:

        S_{ab} = E_{x ~ p}[ psi_a(x) psi_b(x) / p(x) ]
        H_{ab} = E_{x ~ p}[ psi_a(x) (H psi_b)(x) / p(x) ]

    and solves the generalized eigenvalue problem H c = E S c to recover sorted
    individual energies E_0 <= E_1 <= ... <= E_{K-1} and excitation gaps E_k - E_0.

    Args:
        ansatz: MultiStateAnsatz.
        params: Model parameters pytree.
        samples: Single-configuration samples of shape (N, ...) drawn from p(x).
        local_energy_fn: Callable (params, x) -> (K,) single-state local energies.
        log_proposal_prob: Unnormalized log p(x) for each sample, given either as
            an array of shape (N,) or a callable x -> scalar. If None, assumes
            samples were drawn from the natural state mixture
            p(x) proportional to sum_k |psi_k(x)|^2.
        reg: Eigenvalue floor for S in the generalized eigenvalue solve.

    Returns:
        ExcitedStateResult containing sorted energies, excitation gaps,
        eigenvectors, S, and H.
    """
    log_abs, signs = jax.vmap(lambda x: ansatz.eval_states(params, x))(samples)  # (N, K)
    E_L = jax.vmap(lambda x: local_energy_fn(params, x))(samples)  # (N, K)

    if log_proposal_prob is None:
        log_p = jax.scipy.special.logsumexp(2.0 * log_abs, axis=-1)  # (N,)
    elif callable(log_proposal_prob):
        log_p = jax.vmap(log_proposal_prob)(samples)
    else:
        log_p = jnp.asarray(log_proposal_prob)

    # Shift globally so exp(...) stays O(1); global scale cancels in H c = E S c
    log_weighted = log_abs - 0.5 * log_p[:, None]
    global_shift = jax.lax.stop_gradient(jnp.max(log_weighted))
    psi_w = signs * jnp.exp(log_weighted - global_shift)  # (N, K)

    n_samples = samples.shape[0]
    S = (psi_w.T @ psi_w) / n_samples
    H_psi_w = jnp.where(psi_w == 0.0, 0.0, psi_w * E_L)
    H_raw = (psi_w.T @ H_psi_w) / n_samples
    H = 0.5 * (H_raw + H_raw.T)

    energies, eigenvectors = solve_generalized_eigenvalue(H, S, reg=reg)
    excitation_gaps = energies[1:] - energies[0]

    return ExcitedStateResult(
        energies=energies,
        excitation_gaps=excitation_gaps,
        eigenvectors=eigenvectors,
        overlap_matrix=S,
        hamiltonian_matrix=H,
    )


def compute_excited_state_energies_from_nes_samples(
    ansatz: MultiStateAnsatz,
    params: Any,
    X_batch: Array,
    local_energy_fn: Callable[[Any, Array], Array],
) -> ExcitedStateResult:
    """Extract excited-state energies directly from NES-VMC samples X ~ |Psi(X)|^2.

    Uses the identity E_{X ~ |Psi(X)|^2}[ (H M(X)) M(X)^{-1} ] = H S^{-1},
    whose eigenvalues equal the generalized eigenvalues of H c = E S c.

    Args:
        ansatz: MultiStateAnsatz.
        params: Model parameters pytree.
        X_batch: Batch of K-configuration samples of shape (B, K, ...) from |Psi|^2.
        local_energy_fn: Callable (params, x) -> (K,) single-state local energies.

    Returns:
        ExcitedStateResult with sorted real energies and excitation gaps.
    """
    L_batch = jax.vmap(
        lambda X: nes_local_energy_matrix(ansatz, params, X, local_energy_fn)
    )(X_batch)
    L_mean = jnp.mean(L_batch, axis=0)
    eigvals = jnp.linalg.eigvals(L_mean)
    energies = jnp.sort(jnp.real(eigvals))
    excitation_gaps = energies[1:] - energies[0]
    return ExcitedStateResult(
        energies=energies,
        excitation_gaps=excitation_gaps,
        eigenvectors=None,
        overlap_matrix=None,
        hamiltonian_matrix=L_mean,
    )


def metropolis_sample(
    log_prob_fn: Callable[[Array], Array],
    init_positions: Array,
    key: Key,
    *,
    num_steps: int,
    step_size: float = 0.5,
    burn_in: int = 0,
    thin: int = 1,
) -> tuple[Array, Array, Array]:
    """Vectorized random-walk Metropolis-Hastings sampler.

    Args:
        log_prob_fn: Unnormalized log-density function mapping a single state
            of shape ``state_shape`` to a scalar.
        init_positions: Initial chain states of shape ``(num_chains, *state_shape)``.
        key: JAX PRNG key.
        num_steps: Total number of Metropolis-Hastings steps per chain.
        step_size: Gaussian proposal standard deviation.
        burn_in: Number of initial steps to discard.
        thin: Keep every ``thin``-th step after burn-in.

    Returns:
        (samples, final_positions, acceptance_rate) where ``samples`` has shape
        ``(num_kept * num_chains, *state_shape)`` and ``final_positions`` has
        shape ``(num_chains, *state_shape)``.
    """
    init_log_probs = jax.vmap(log_prob_fn)(init_positions)

    def mh_step(carry, step_key):
        pos, lp = carry
        prop_key, acc_key = jax.random.split(step_key)
        proposal = pos + step_size * jax.random.normal(prop_key, shape=pos.shape, dtype=pos.dtype)
        prop_lp = jax.vmap(log_prob_fn)(proposal)

        log_alpha = prop_lp - lp
        log_u = jnp.log(jax.random.uniform(acc_key, shape=lp.shape, dtype=lp.dtype))
        accept = jnp.isfinite(prop_lp) & (log_u < log_alpha)

        expand_shape = (accept.shape[0],) + (1,) * (pos.ndim - 1)
        new_pos = jnp.where(accept.reshape(expand_shape), proposal, pos)
        new_lp = jnp.where(accept, prop_lp, lp)
        return (new_pos, new_lp), (new_pos, accept.astype(jnp.float32))

    step_keys = jax.random.split(key, num_steps)
    (final_pos, _), (trajectory, accepts) = jax.lax.scan(
        mh_step, (init_positions, init_log_probs), step_keys
    )

    kept = trajectory[burn_in::thin]
    samples = kept.reshape((-1,) + init_positions.shape[1:])
    acceptance_rate = jnp.mean(accepts)
    return samples, final_pos, acceptance_rate
