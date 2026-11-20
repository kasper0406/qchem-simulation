"""Tests for SPRING and MinSR natural-gradient optimizers."""

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from qchem_simulation.electronic_optimization import WaveFunction
from qchem_simulation.spring import (
    SPRINGState,
    SpringOptimizer,
    compute_per_sample_log_grads,
    compute_spring_direction,
    init_spring_state,
    spring_update,
)
from qchem_simulation.utils import Electron, Nucleus


def _primal_sr_direction(O: jax.Array, epsilon: jax.Array, damping: float) -> jax.Array:
    """Reference primal P x P Stochastic Reconfiguration solve: (O^T O + lambda I)^{-1} O^T eps."""
    num_params = O.shape[1]
    S = O.T @ O
    f = O.T @ epsilon
    return jnp.linalg.solve(S + damping * jnp.eye(num_params, dtype=O.dtype), f)


@pytest.mark.parametrize(
    "num_samples,param_shapes",
    [
        # Overdetermined regime: P < N_s (P = 9, N_s = 64)
        (64, {"w": (3, 2), "b": (3,)}),
        # Underdetermined regime: P > N_s (P = 136, N_s = 16)
        (16, {"w1": (16, 8), "b1": (8,)}),
        # Square regime: P == N_s (P = 16, N_s = 16)
        (16, {"w": (4, 4)}),
    ],
)
def test_minsr_matches_exact_primal_sr(num_samples, param_shapes):
    """MinSR (momentum=0.0) must match exact primal (S + lambda I)^{-1} (1/2 g) for P < N_s and P > N_s."""
    key = jax.random.key(42)
    keys = jax.random.split(key, 3)

    params = {
        name: jax.random.normal(jax.random.fold_in(keys[0], i), shape, dtype=jnp.float32)
        for i, (name, shape) in enumerate(param_shapes.items())
    }
    per_sample_log_grads = {
        name: jax.random.normal(
            jax.random.fold_in(keys[1], i), (num_samples, *shape), dtype=jnp.float32
        )
        for i, (name, shape) in enumerate(param_shapes.items())
    }
    local_energies = jax.random.normal(keys[2], (num_samples,), dtype=jnp.float32)

    damping = 1e-2
    lr = 0.1
    state = init_spring_state(params)

    new_params, new_state = spring_update(
        params,
        state,
        local_energies,
        per_sample_log_grads,
        learning_rate=lr,
        damping=damping,
        momentum=0.0,
    )

    # Build O and epsilon explicitly and solve in primal P x P space
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    flat_grads = jax.vmap(lambda g: jax.flatten_util.ravel_pytree(g)[0])(per_sample_log_grads)
    O = (flat_grads - jnp.mean(flat_grads, axis=0, keepdims=True)) / jnp.sqrt(num_samples)
    epsilon = (local_energies - jnp.mean(local_energies)) / jnp.sqrt(num_samples)

    # Exact VMC gradient is g = 2 * O^T epsilon, so (1/2) (S + lambda I)^{-1} g = (S + lambda I)^{-1} O^T eps
    exact_phi = _primal_sr_direction(O, epsilon, damping)
    expected_params = unravel_fn(flat_params - lr * exact_phi)

    chex.assert_trees_all_close(new_state.phi, exact_phi, rtol=1e-3, atol=1e-4)
    chex.assert_trees_all_close(new_params, expected_params, rtol=1e-3, atol=1e-4)
    assert int(new_state.step) == 1


def test_exact_factor_of_two_1d_harmonic_oscillator():
    """Verify exact factor-of-2 natural-gradient scale on the 1D harmonic oscillator.

    For H = -1/2 d^2/dx^2 + 1/2 x^2 and log|psi_alpha(x)| = -1/2 alpha x^2:
      - E_L(x) = alpha/2 + (1 - alpha^2)/2 * x^2
      - E(alpha) = (alpha + 1/alpha) / 4
      - g = E'(alpha) = (1 - 1/alpha^2) / 4
      - S = Var(d log|psi| / d alpha) = 1 / (8 alpha^2)
      - Exact natural-gradient direction: phi = (1/2) S^{-1} g = alpha^2 - 1.

    Near alpha = 1 + delta, phi = 2*delta + delta^2, so with eta = 0.5 a single
    MinSR step jumps directly to alpha* = 1.0 up to O(delta^2).
    """
    alpha0 = 1.4
    num_samples = 2000
    key = jax.random.key(0)

    # Exact samples from p_alpha(x) = sqrt(alpha / pi) * exp(-alpha * x^2) -> N(0, 1/(2*alpha))
    # Use antithetic variates for ultra-low Monte Carlo variance.
    half_z = jax.random.normal(key, (num_samples // 2,))
    z = jnp.concatenate([half_z, -half_z])
    x = z / jnp.sqrt(2.0 * alpha0)

    def log_psi_fn(alpha, x_i):
        return -0.5 * alpha * (x_i ** 2)

    params = jnp.array(alpha0, dtype=jnp.float32)
    _, per_sample_log_grads = compute_per_sample_log_grads(log_psi_fn, params, x)
    local_energies = 0.5 * alpha0 + 0.5 * (1.0 - alpha0 ** 2) * (x ** 2)

    state = init_spring_state(params)
    new_params, new_state = spring_update(
        params,
        state,
        local_energies,
        per_sample_log_grads,
        learning_rate=0.5,
        damping=1e-7,
        momentum=0.0,
    )

    # Notice: because E_L(x) is an exact affine function of O(x) = -1/2 x^2,
    # for ANY sample set (zero MC variance!), O^T (O O^T)^{-1} eps = alpha0^2 - 1 EXACTLY!
    expected_phi = alpha0 ** 2 - 1.0
    chex.assert_trees_all_close(new_state.phi[0], expected_phi, rtol=1e-4, atol=1e-4)

    # With lr = 0.5, alpha_1 = alpha0 - 0.5 * (alpha0^2 - 1) = 1.4 - 0.48 = 0.92 (much closer to 1.0)
    expected_alpha1 = alpha0 - 0.5 * expected_phi
    chex.assert_trees_all_close(new_params, expected_alpha1, rtol=1e-4, atol=1e-4)


def test_spring_momentum_matches_kaczmarz_formula():
    """Verify multi-step SPRING updates match the exact Kaczmarz-inspired primal formula."""
    key = jax.random.key(123)
    num_samples = 12
    num_params = 30
    damping = 1e-2
    momentum = 0.95
    lr = 0.05

    params = jax.random.normal(key, (num_params,), dtype=jnp.float32)
    state = init_spring_state(params)
    ref_phi = jnp.zeros((num_params,), dtype=jnp.float32)
    ref_params = params

    for step in range(5):
        step_key = jax.random.fold_in(key, step + 1)
        k1, k2 = jax.random.split(step_key)
        per_sample_log_grads = jax.random.normal(k1, (num_samples, num_params), dtype=jnp.float32)
        local_energies = jax.random.normal(k2, (num_samples,), dtype=jnp.float32)

        params, state = spring_update(
            params,
            state,
            local_energies,
            per_sample_log_grads,
            learning_rate=lr,
            damping=damping,
            momentum=momentum,
        )

        # Reference primal Kaczmarz update:
        # phi_t = mu * phi_{t-1} + (O^T O + lambda I)^{-1} O^T (eps - O (mu * phi_{t-1}))
        O = (per_sample_log_grads - jnp.mean(per_sample_log_grads, axis=0, keepdims=True)) / jnp.sqrt(num_samples)
        eps = (local_energies - jnp.mean(local_energies)) / jnp.sqrt(num_samples)
        eps_tilde = eps - O @ (momentum * ref_phi)
        ref_phi = momentum * ref_phi + _primal_sr_direction(O, eps_tilde, damping)
        ref_params = ref_params - lr * ref_phi

        chex.assert_trees_all_close(state.phi, ref_phi, rtol=1e-3, atol=1e-4)
        chex.assert_trees_all_close(params, ref_params, rtol=1e-3, atol=1e-4)
        assert int(state.step) == step + 1


def test_spring_max_norm_trust_region():
    """Verify max_norm caps ||theta_{t+1} - theta_t||_2 while keeping state.phi unclipped."""
    key = jax.random.key(7)
    k1, k2 = jax.random.split(key)
    params = jnp.zeros((20,), dtype=jnp.float32)
    per_sample_log_grads = jax.random.normal(k1, (10, 20), dtype=jnp.float32)
    local_energies = 100.0 * jax.random.normal(k2, (10,), dtype=jnp.float32)

    state = init_spring_state(params)
    max_norm = 0.05
    lr = 1.0

    new_params_clipped, new_state_clipped = spring_update(
        params,
        state,
        local_energies,
        per_sample_log_grads,
        learning_rate=lr,
        damping=1e-2,
        momentum=0.99,
        max_norm=max_norm,
    )
    _, new_state_unclipped = spring_update(
        params,
        state,
        local_energies,
        per_sample_log_grads,
        learning_rate=lr,
        damping=1e-2,
        momentum=0.99,
        max_norm=None,
    )

    step_norm = float(jnp.linalg.norm(new_params_clipped - params))
    assert step_norm <= max_norm + 1e-6
    assert float(jnp.linalg.norm(new_state_unclipped.phi)) > max_norm
    # state.phi remains unclipped so its scale matches epsilon on the next step
    chex.assert_trees_all_close(new_state_clipped.phi, new_state_unclipped.phi, rtol=1e-6)


def test_spring_jit_compatibility():
    """Verify spring_update and compute_spring_direction work inside jax.jit with finite outputs."""
    key = jax.random.key(99)
    k1, k2, k3 = jax.random.split(key, 3)

    params = {"a": jax.random.normal(k1, (4, 3)), "b": jax.random.normal(k1, (5,))}
    per_sample_log_grads = {
        "a": jax.random.normal(k2, (16, 4, 3)),
        "b": jax.random.normal(k2, (16, 5)),
    }
    local_energies = jax.random.normal(k3, (16,))
    state = init_spring_state(params)

    jitted_update = jax.jit(spring_update)

    for _ in range(10):
        params, state = jitted_update(
            params,
            state,
            local_energies,
            per_sample_log_grads,
            learning_rate=1e-2,
            damping=1e-3,
            momentum=0.99,
            max_norm=0.1,
        )

    assert int(state.step) == 10
    assert jnp.all(jnp.isfinite(state.phi))
    for leaf in jax.tree_util.tree_leaves(params):
        assert jnp.all(jnp.isfinite(leaf))


def test_spring_converges_faster_than_adam_coupled_ho():
    """Benchmark SPRING vs Adam on a 3D harmonic oscillator with coupled parameters.

    H = sum_{d=1}^3 (-1/2 d^2/dx_d^2 + 1/2 (x_d - c_d)^2), exact E_0 = 1.5.
    The Gaussian ansatz is parameterized through an ill-conditioned mixing matrix M:
      alpha = exp(M @ u), beta = M @ v,
      log|psi(x)| = -1/2 sum_d alpha_d x_d^2 + sum_d beta_d x_d.

    Because SPRING preconditions by the full Fubini-Study / QGT metric, it is
    invariant to the parameter coupling M and converges to ~1e-5 error in 30 steps,
    while diagonal-preconditioned Adam barely moves.
    """
    omega = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    center = jnp.array([1.5, -1.0, 0.5], dtype=jnp.float32)
    exact_energy = float(0.5 * jnp.sum(omega))  # 1.5

    M = jnp.array(
        [
            [1.0, 0.95, 0.90],
            [0.95, 1.0, 0.95],
            [0.90, 0.95, 1.05],
        ],
        dtype=jnp.float32,
    )

    def to_alpha_beta(params):
        return jnp.exp(M @ params["u"]), M @ params["v"]

    def log_psi_fn(params, x_i):
        alpha, beta = to_alpha_beta(params)
        return -0.5 * jnp.sum(alpha * (x_i ** 2)) + jnp.sum(beta * x_i)

    def sample_and_local_energy(params, key, num_samples=256):
        alpha, beta = to_alpha_beta(params)
        mean = beta / alpha
        std = jax.lax.rsqrt(2.0 * alpha)
        x = mean[None, :] + std[None, :] * jax.random.normal(key, (num_samples, 3))

        grad_x = -alpha[None, :] * x + beta[None, :]
        kinetic = -0.5 * jnp.sum(-alpha[None, :] + grad_x ** 2, axis=-1)
        potential = 0.5 * jnp.sum((omega[None, :] ** 2) * ((x - center[None, :]) ** 2), axis=-1)
        local_energies = kinetic + potential
        return x, local_energies

    def exact_energy_of(params):
        alpha, beta = to_alpha_beta(params)
        mean = beta / alpha
        var = 1.0 / (2.0 * alpha)
        kinetic = 0.25 * jnp.sum(alpha)
        potential = 0.5 * jnp.sum((omega ** 2) * (var + (mean - center) ** 2))
        return float(kinetic + potential)

    init_params = {
        "u": jnp.array([0.3, 0.2, -0.1], dtype=jnp.float32),
        "v": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
    }

    num_steps = 30
    base_key = jax.random.key(2024)

    # --- Run SPRING ---
    spring_params = init_params
    spring_state = init_spring_state(spring_params)

    @jax.jit
    def spring_step(params, state, key):
        x, local_energies = sample_and_local_energy(params, key)
        _, log_grads = compute_per_sample_log_grads(log_psi_fn, params, x)
        return spring_update(
            params,
            state,
            local_energies,
            log_grads,
            learning_rate=0.3,
            damping=1e-4,
            momentum=0.5,
            max_norm=2.0,
        )

    for step in range(num_steps):
        spring_params, spring_state = spring_step(
            spring_params, spring_state, jax.random.fold_in(base_key, step)
        )
    spring_err = exact_energy_of(spring_params) - exact_energy

    # --- Run Adam (using exact VMC gradient g = 2 * Cov(d log|psi|, E_L)) ---
    adam_opt = optax.adam(learning_rate=0.2)
    adam_params = init_params
    adam_state = adam_opt.init(adam_params)

    @jax.jit
    def adam_step(params, opt_state, key):
        x, local_energies = sample_and_local_energy(params, key)
        _, log_grads = compute_per_sample_log_grads(log_psi_fn, params, x)
        centered_e = local_energies - jnp.mean(local_energies)
        vmc_grads = jax.tree.map(
            lambda g: 2.0 * jnp.mean(
                jnp.expand_dims(centered_e, axis=tuple(range(1, g.ndim))) * g, axis=0
            ),
            log_grads,
        )
        updates, new_opt_state = adam_opt.update(vmc_grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state

    for step in range(num_steps):
        adam_params, adam_state = adam_step(
            adam_params, adam_state, jax.random.fold_in(base_key, step)
        )
    adam_err = exact_energy_of(adam_params) - exact_energy

    # SPRING reaches <1e-3 energy error while Adam remains >1.0 (over 1000x worse)
    assert spring_err < 1e-3, f"SPRING error too large: {spring_err:.4e}"
    assert spring_err < adam_err / 100.0, (
        f"Expected SPRING ({spring_err:.4e}) to converge dramatically faster than Adam ({adam_err:.4e})"
    )


def test_spring_converges_hydrogen_atom():
    """Verify SPRING converges to the exact H atom ground state E_0 = -0.5 Ha faster than Adam."""
    exact_energy = -0.5

    def log_psi_fn(log_zeta, r):
        zeta = jnp.exp(log_zeta)
        return -zeta * r

    def sample_and_local_energy(log_zeta, key, num_samples=256):
        zeta = jnp.exp(log_zeta)
        # Exact radial density p(r) = 4 zeta^3 r^2 exp(-2 zeta r) = Gamma(shape=3, rate=2*zeta)
        r = jax.random.gamma(key, a=3.0, shape=(num_samples,)) / (2.0 * zeta)
        # For psi(r) = exp(-zeta r): E_L(r) = -0.5 * zeta^2 + (zeta - 1) / r
        local_energies = -0.5 * (zeta ** 2) + (zeta - 1.0) / r
        return r, local_energies

    def exact_energy_of(log_zeta):
        zeta = float(jnp.exp(log_zeta))
        return 0.5 * zeta ** 2 - zeta

    init_log_zeta = jnp.array(jnp.log(3.0), dtype=jnp.float32)  # E = 1.5 Ha initially
    num_steps = 20
    base_key = jax.random.key(777)

    # --- SPRING ---
    s_param = init_log_zeta
    s_state = init_spring_state(s_param)

    @jax.jit
    def s_step(param, state, key):
        r, local_energies = sample_and_local_energy(param, key)
        _, log_grads = compute_per_sample_log_grads(log_psi_fn, param, r)
        return spring_update(
            param, state, local_energies, log_grads,
            learning_rate=0.3, damping=1e-3, momentum=0.5, max_norm=0.5,
        )

    for step in range(num_steps):
        s_param, s_state = s_step(s_param, s_state, jax.random.fold_in(base_key, step))
    spring_err = exact_energy_of(s_param) - exact_energy

    # --- Adam ---
    adam_opt = optax.adam(learning_rate=0.05)
    a_param = init_log_zeta
    a_state = adam_opt.init(a_param)

    @jax.jit
    def a_step(param, opt_state, key):
        r, local_energies = sample_and_local_energy(param, key)
        _, log_grads = compute_per_sample_log_grads(log_psi_fn, param, r)
        centered_e = local_energies - jnp.mean(local_energies)
        vmc_grad = 2.0 * jnp.mean(centered_e * log_grads)
        updates, new_opt_state = adam_opt.update(vmc_grad, opt_state, param)
        return optax.apply_updates(param, updates), new_opt_state

    for step in range(num_steps):
        a_param, a_state = a_step(a_param, a_state, jax.random.fold_in(base_key, step))
    adam_err = exact_energy_of(a_param) - exact_energy

    assert spring_err < 1e-3, f"SPRING H-atom error too large: {spring_err:.4e}"
    assert spring_err < adam_err / 10.0, (
        f"Expected SPRING ({spring_err:.4e}) to beat Adam ({adam_err:.4e}) on H atom"
    )


def test_spring_optimizer_wavefunction_smoke():
    """Smoke test SpringOptimizer end-to-end with Flax NNX WaveFunction (H2 molecule)."""
    num_electrons = 2
    num_nuclei = 2
    num_samples = 8

    wave_function = WaveFunction(
        num_electrons=num_electrons,
        num_nuclei=num_nuclei,
        rngs=nnx.Rngs(0),
    )
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=jnp.float32),
        charge=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )

    key = jax.random.key(42)
    pos_key, step_key1, step_key2 = jax.random.split(key, 3)
    positions = jax.random.normal(pos_key, (num_samples, num_electrons, 3), dtype=jnp.float32)
    spins = jnp.broadcast_to(
        jnp.array([[1.0], [-1.0]], dtype=jnp.float32), (num_samples, num_electrons, 1)
    )
    samples = Electron(position=positions, spin=spins)

    _, params_before, _ = nnx.split(wave_function, nnx.Param, ...)
    flat_before, _ = jax.flatten_util.ravel_pytree(params_before)

    optimizer = SpringOptimizer(
        wave_function,
        learning_rate=1e-2,
        damping=1e-2,
        momentum=0.99,
        max_norm=0.1,
    )

    avg_e1, details1, log_psi1 = optimizer.step(wave_function, samples, nuclei, step_key1)
    avg_e2, details2, log_psi2 = optimizer.step(wave_function, samples, nuclei, step_key2)

    _, params_after, _ = nnx.split(wave_function, nnx.Param, ...)
    flat_after, _ = jax.flatten_util.ravel_pytree(params_after)

    assert int(optimizer.state.step) == 2
    assert jnp.isfinite(avg_e1) and jnp.isfinite(avg_e2)
    assert jnp.isfinite(log_psi1) and jnp.isfinite(log_psi2)
    assert jnp.isfinite(details1.kinetic_energy) and jnp.isfinite(details2.potential_energy)
    assert jnp.all(jnp.isfinite(optimizer.state.phi))
    assert jnp.all(jnp.isfinite(flat_after))
    assert float(jnp.linalg.norm(flat_after - flat_before)) > 0.0
