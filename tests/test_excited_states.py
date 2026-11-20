"""Unit tests for Natural Excited States VMC (NES-VMC)."""

import jax
import jax.numpy as jnp
import optax
import pytest

from qchem_simulation.excited_states import (
    MultiStateAnsatz,
    compute_excited_state_energies,
    compute_excited_state_energies_from_nes_samples,
    make_schrodinger_local_energy_fn,
    metropolis_sample,
    nes_local_energy,
    nes_vmc_step,
)


# ---------------------------------------------------------------------------
# Analytical helper wavefunctions
# ---------------------------------------------------------------------------

def _ho_1d_potential(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum(x ** 2)


def _ho_1d_raw_states(x: jax.Array, num_states: int) -> jax.Array:
    """Exact unnormalized 1D Harmonic Oscillator eigenstates psi_0, psi_1, psi_2."""
    x0 = x[0]
    env = jnp.exp(-0.5 * x0 ** 2)
    states = [
        env,                      # E_0 = 0.5
        x0 * env,                 # E_1 = 1.5
        (2.0 * x0 ** 2 - 1.0) * env,  # E_2 = 2.5
    ]
    return jnp.stack(states[:num_states], axis=0)


def _signed_to_log_sign(vals: jax.Array) -> tuple[jax.Array, jax.Array]:
    signs = jnp.where(vals >= 0.0, 1.0, -1.0)
    log_abs = jnp.log(jnp.abs(vals))
    return log_abs, signs


def _make_ho_1d_ansatz(num_states: int, mix_matrix: jax.Array | None = None) -> MultiStateAnsatz:
    def single_state_fn(_params, x):
        raw = _ho_1d_raw_states(x, num_states)
        if mix_matrix is not None:
            raw = mix_matrix @ raw
        return _signed_to_log_sign(raw)

    return MultiStateAnsatz(num_states=num_states, single_state_fn=single_state_fn)


# ---------------------------------------------------------------------------
# 1. Exact analytical local energy tests (1D & 3D Harmonic Oscillator)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_states,expected_sum", [(2, 2.0), (3, 4.5)])
def test_ho_1d_exact_nes_local_energy(num_states: int, expected_sum: float):
    """Exact 1D HO eigenstates must yield constant nes_local_energy = sum(E_k)."""
    ansatz = _make_ho_1d_ansatz(num_states)
    local_energy_fn = make_schrodinger_local_energy_fn(ansatz, _ho_1d_potential)

    key = jax.random.key(42 + num_states)
    # Batch of random K-configuration points in 1D: shape (16, K, 1)
    X_batch = jax.random.normal(key, shape=(16, num_states, 1)) * 1.5
    energies = jax.vmap(lambda X: nes_local_energy(ansatz, None, X, local_energy_fn))(X_batch)

    assert jnp.all(jnp.isfinite(energies))
    assert jnp.allclose(energies, expected_sum, atol=1e-4), (
        f"Expected constant {expected_sum}, got {energies}"
    )


def test_ho_1d_non_orthogonal_rotation_invariance():
    """Arbitrary invertible non-orthogonal linear combinations must give exact sum(E_k)."""
    # K = 2 non-orthogonal mixing
    mix_2 = jnp.array([[1.0, 0.85], [-0.45, 1.3]], dtype=jnp.float32)
    ansatz_2 = _make_ho_1d_ansatz(2, mix_matrix=mix_2)
    le_fn_2 = make_schrodinger_local_energy_fn(ansatz_2, _ho_1d_potential)

    X_batch_2 = jax.random.normal(jax.random.key(101), shape=(12, 2, 1)) * 1.3
    energies_2 = jax.vmap(lambda X: nes_local_energy(ansatz_2, None, X, le_fn_2))(X_batch_2)
    assert jnp.allclose(energies_2, 2.0, atol=1e-4)

    # K = 3 non-orthogonal mixing
    mix_3 = jnp.array(
        [
            [1.0, 0.6, -0.3],
            [-0.5, 1.2, 0.7],
            [0.4, -0.8, 1.1],
        ],
        dtype=jnp.float32,
    )
    ansatz_3 = _make_ho_1d_ansatz(3, mix_matrix=mix_3)
    le_fn_3 = make_schrodinger_local_energy_fn(ansatz_3, _ho_1d_potential)

    X_batch_3 = jax.random.normal(jax.random.key(202), shape=(12, 3, 1)) * 1.3
    energies_3 = jax.vmap(lambda X: nes_local_energy(ansatz_3, None, X, le_fn_3))(X_batch_3)
    assert jnp.allclose(energies_3, 4.5, atol=1e-4)

    # Verify local energy matrix directly recovers individual eigenenergies [0.5, 1.5, 2.5]
    result = compute_excited_state_energies_from_nes_samples(ansatz_3, None, X_batch_3, le_fn_3)
    assert jnp.allclose(result.energies, jnp.array([0.5, 1.5, 2.5]), atol=1e-4)
    assert jnp.allclose(result.excitation_gaps, jnp.array([1.0, 2.0]), atol=1e-4)


def test_ho_3d_exact_nes_local_energy():
    """Exact 3D Harmonic Oscillator (ground E0=1.5 + first excited E1=2.5 => sum=4.0)."""
    mix = jnp.array([[1.0, 0.5], [-0.7, 1.1]], dtype=jnp.float32)

    def single_state_fn(_params, r):
        r2 = jnp.sum(r ** 2)
        env = jnp.exp(-0.5 * r2)
        raw = jnp.array([env, r[0] * env])
        mixed = mix @ raw
        return _signed_to_log_sign(mixed)

    ansatz = MultiStateAnsatz(num_states=2, single_state_fn=single_state_fn)
    le_fn = make_schrodinger_local_energy_fn(ansatz, lambda r: 0.5 * jnp.sum(r ** 2))

    X_batch = jax.random.normal(jax.random.key(303), shape=(12, 2, 3)) * 1.2
    energies = jax.vmap(lambda X: nes_local_energy(ansatz, None, X, le_fn))(X_batch)
    assert jnp.allclose(energies, 4.0, atol=1e-4)


def test_log_space_column_shift_stability():
    """Extreme log-scale differences across configurations must not overflow or underflow."""
    # Evaluate 1D HO states at large |x| where exp(-x^2/2) would underflow to 0 in float32
    def psi_0(_params, x):
        return -0.5 * jnp.sum(x ** 2), jnp.array(1.0)

    def psi_1(_params, x):
        x0 = x[0]
        return jnp.log(jnp.abs(x0)) - 0.5 * jnp.sum(x ** 2), jnp.where(x0 >= 0.0, 1.0, -1.0)

    ansatz = MultiStateAnsatz.from_state_fns([psi_0, psi_1])
    le_fn = make_schrodinger_local_energy_fn(ansatz, _ho_1d_potential)

    # At x = 15.0, exp(-0.5 * 225) = exp(-112.5) ~ 1.4e-49 which underflows float32 (min ~1.2e-38)!
    X = jnp.array([[15.0], [-14.0]], dtype=jnp.float32)
    log_abs_psi, sign_psi = ansatz(None, X)
    assert jnp.isfinite(log_abs_psi)
    assert jnp.abs(sign_psi) == 1.0

    # Local energy must still be exact = 2.0 despite extreme log-magnitudes
    e_loc = nes_local_energy(ansatz, None, X, le_fn)
    assert jnp.isfinite(e_loc)
    assert jnp.allclose(e_loc, 2.0, atol=1e-3)


def test_subspace_diagonalization_on_non_orthogonal_exact_states():
    """compute_excited_state_energies recovers exact E_0, E_1 from non-orthogonal basis."""
    mix = jnp.array([[1.0, 0.9], [0.4, -1.2]], dtype=jnp.float32)
    ansatz = _make_ho_1d_ansatz(2, mix_matrix=mix)
    le_fn = make_schrodinger_local_energy_fn(ansatz, _ho_1d_potential)

    # Sample single configurations from the state mixture p(x) ~ |psi_0|^2 + |psi_1|^2
    init_pos = jax.random.normal(jax.random.key(11), shape=(64, 1))
    samples, _, acc = metropolis_sample(
        lambda x: ansatz.mixture_log_prob(None, x),
        init_pos,
        jax.random.key(12),
        num_steps=80,
        step_size=1.0,
        burn_in=30,
        thin=2,
    )
    assert acc > 0.2

    res = compute_excited_state_energies(ansatz, None, samples, le_fn)
    assert res.overlap_matrix is not None
    assert res.hamiltonian_matrix is not None
    assert jnp.allclose(res.energies, jnp.array([0.5, 1.5]), atol=5e-3)
    assert jnp.allclose(res.excitation_gaps, jnp.array([1.0]), atol=5e-3)


# ---------------------------------------------------------------------------
# 2. End-to-end NES-VMC optimization (1D Harmonic Oscillator, K=2)
# ---------------------------------------------------------------------------

def _init_mlp_2state_params(key: jax.Array, hidden_dim: int = 16) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (1, hidden_dim)) * 0.8,
        "b1": jax.random.normal(k2, (hidden_dim,)) * 0.2,
        "w2": jax.random.normal(k3, (hidden_dim, 2)) * 0.5,
        "b2": jnp.zeros((2,)),
        # Envelope width parameter (softplus -> alpha; initialized away from exact 0.5)
        "raw_alpha": jnp.array([-0.5, -0.8]),
    }


def _make_neural_2state_ansatz() -> MultiStateAnsatz:
    def single_state_fn(params: dict, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(x @ params["w1"] + params["b1"])
        out = h @ params["w2"] + params["b2"]  # shape (2,)
        alpha = jax.nn.softplus(params["raw_alpha"])  # positive envelope exponent
        r2 = jnp.sum(x ** 2)
        log_abs = jnp.log(jnp.abs(out) + 1e-12) - alpha * r2
        sign = jnp.where(out >= 0.0, 1.0, -1.0)
        return log_abs, sign

    return MultiStateAnsatz(num_states=2, single_state_fn=single_state_fn)


def test_end_to_end_nes_vmc_ho_1d():
    """Train a neural 2-state ansatz from scratch via NES-VMC on 1D Harmonic Oscillator."""
    ansatz = _make_neural_2state_ansatz()
    le_fn = make_schrodinger_local_energy_fn(ansatz, _ho_1d_potential)

    key = jax.random.key(0)
    k_init, k_walk, k_train, k_eval = jax.random.split(key, 4)

    params = _init_mlp_2state_params(k_init, hidden_dim=16)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.5e-2))
    opt_state = optimizer.init(params)

    num_chains = 128
    walkers = jax.random.normal(k_walk, shape=(num_chains, 2, 1))

    @jax.jit
    def train_step(params, opt_state, walkers, step_key):
        samples, new_walkers, acc = metropolis_sample(
            lambda X: ansatz.log_prob(params, X),
            walkers,
            step_key,
            num_steps=15,
            step_size=0.9,
            burn_in=10,
            thin=1,
        )
        new_params, new_opt_state, metrics = nes_vmc_step(
            ansatz,
            params,
            opt_state,
            optimizer,
            samples,
            le_fn,
            clip_factor=5.0,
        )
        return new_params, new_opt_state, new_walkers, metrics, acc

    # Initial burn-in
    _, walkers, _ = metropolis_sample(
        lambda X: ansatz.log_prob(params, X),
        walkers,
        jax.random.key(999),
        num_steps=50,
        step_size=0.9,
        burn_in=49,
    )

    step_keys = jax.random.split(k_train, 250)
    energy_history = []
    for sk in step_keys:
        params, opt_state, walkers, metrics, _acc = train_step(params, opt_state, walkers, sk)
        energy_history.append(float(metrics["total_energy"]))

    # Final total energy should converge close to E_0 + E_1 = 2.0
    final_total_energy = float(jnp.mean(jnp.array(energy_history[-25:])))
    assert abs(final_total_energy - 2.0) < 0.05, (
        f"Total energy did not converge to 2.0: got {final_total_energy:.4f}"
    )

    # Subspace diagonalization on single-configuration mixture samples
    k_eval_init, k_eval_mcmc = jax.random.split(k_eval)
    eval_init = jax.random.normal(k_eval_init, shape=(128, 1))
    eval_samples, _, _ = metropolis_sample(
        lambda x: ansatz.mixture_log_prob(params, x),
        eval_init,
        k_eval_mcmc,
        num_steps=150,
        step_size=0.9,
        burn_in=50,
        thin=2,
    )

    result = compute_excited_state_energies(ansatz, params, eval_samples, le_fn)
    e0, e1 = float(result.energies[0]), float(result.energies[1])
    gap = float(result.excitation_gaps[0])

    assert abs(e0 - 0.5) < 0.04, f"Expected E_0 ~ 0.5, got {e0:.4f}"
    assert abs(e1 - 1.5) < 0.04, f"Expected E_1 ~ 1.5, got {e1:.4f}"
    assert abs(gap - 1.0) < 0.05, f"Expected excitation gap ~ 1.0, got {gap:.4f}"


# ---------------------------------------------------------------------------
# 3. Quantum chemistry smoke test (Hydrogen 1s + 2s, K=2)
# ---------------------------------------------------------------------------

def test_hydrogen_1s_2s_exact_nes_local_energy():
    """Exact Hydrogen 1s (E=-0.5 Ha) and 2s (E=-0.125 Ha) yield exact sum -0.625 Ha."""

    def coulomb_potential(r: jax.Array) -> jax.Array:
        return -1.0 / jnp.linalg.norm(r)

    def psi_1s(_params, r):
        dist = jnp.linalg.norm(r)
        return -dist, jnp.array(1.0)

    def psi_2s(_params, r):
        dist = jnp.linalg.norm(r)
        val = 2.0 - dist
        return jnp.log(jnp.abs(val)) - 0.5 * dist, jnp.where(val >= 0.0, 1.0, -1.0)

    ansatz = MultiStateAnsatz.from_state_fns([psi_1s, psi_2s])
    le_fn = make_schrodinger_local_energy_fn(ansatz, coulomb_potential)

    # Generate 3D configurations away from origin and r = 2 node
    key = jax.random.key(777)
    X_batch = jax.random.normal(key, shape=(16, 2, 3)) * 1.5
    energies = jax.vmap(lambda X: nes_local_energy(ansatz, None, X, le_fn))(X_batch)

    assert jnp.all(jnp.isfinite(energies))
    assert jnp.allclose(energies, -0.625, atol=1e-4), (
        f"Expected constant -0.625 Ha, got {energies}"
    )

    # Also verify non-orthogonal linear combination of 1s and 2s
    mix = jnp.array([[1.0, 0.7], [-0.5, 1.3]], dtype=jnp.float32)

    def mixed_h_states(_params, r):
        dist = jnp.linalg.norm(r)
        raw = jnp.array([jnp.exp(-dist), (2.0 - dist) * jnp.exp(-0.5 * dist)])
        return _signed_to_log_sign(mix @ raw)

    mixed_ansatz = MultiStateAnsatz(num_states=2, single_state_fn=mixed_h_states)
    mixed_le_fn = make_schrodinger_local_energy_fn(mixed_ansatz, coulomb_potential)
    mixed_energies = jax.vmap(
        lambda X: nes_local_energy(mixed_ansatz, None, X, mixed_le_fn)
    )(X_batch)
    assert jnp.allclose(mixed_energies, -0.625, atol=1e-4)

    # Verify local energy matrix recovers E_1s = -0.5, E_2s = -0.125, gap = 0.375 Ha
    res = compute_excited_state_energies_from_nes_samples(
        mixed_ansatz, None, X_batch, mixed_le_fn
    )
    assert jnp.allclose(res.energies, jnp.array([-0.5, -0.125]), atol=1e-4)
    assert jnp.allclose(res.excitation_gaps, jnp.array([0.375]), atol=1e-4)


def test_subspace_energy_is_invariant_to_state_rescaling_and_flags_rank_loss():
    from qchem_simulation.excited_states import solve_generalized_eigenvalue

    overlap = jnp.diag(jnp.array([1., 1e-12]))
    hamiltonian = jnp.diag(jnp.array([0.5, 1.5e-12]))
    energies, c = solve_generalized_eigenvalue(hamiltonian, overlap)
    assert jnp.allclose(energies, jnp.array([0.5, 1.5]))
    assert jnp.allclose(c.T @ overlap @ c, jnp.eye(2))
    energies, c = solve_generalized_eigenvalue(jnp.ones((2, 2)), jnp.ones((2, 2)))
    assert jnp.all(jnp.isnan(energies)) and jnp.all(jnp.isnan(c))


def test_signed_action_at_individual_state_node():
    def amplitudes(p, x, shift):
        t = x[0]
        return jnp.array([1., t + t*t + p]) * jnp.exp(-0.5*t*t - shift)

    def log_sign(p, x):
        return _signed_to_log_sign(amplitudes(p, x, 0.))

    ansatz = MultiStateAnsatz(2, log_sign, scaled_amplitudes=amplitudes)
    action = make_schrodinger_local_energy_fn(ansatz, _ho_1d_potential)
    x = jnp.array([[0.], [1.]])
    assert jnp.allclose(nes_local_energy(ansatz, 0., x, action), 2.5)
    assert jnp.allclose(jax.grad(lambda p: ansatz.log_prob(p, x))(0.), 0.)
    result = compute_excited_state_energies(ansatz, 0., x, action, log_proposal_prob=jnp.zeros(2))
    assert jnp.all(jnp.isfinite(result.energies))
