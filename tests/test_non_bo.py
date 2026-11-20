"""Unit and physics tests for Non-Born-Oppenheimer (Quantum Nuclei + Electrons) VMC."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from qchem_simulation.non_bo import (
    DEUTERON_MASS,
    ELECTRON_MASS,
    PROTON_MASS,
    NonBOWaveFunction,
    QuantumSystem,
    calculate_non_bo_kinetic_energy,
    calculate_per_particle_kinetic_energy,
    center_of_mass,
    make_h2_plus,
    make_hydrogen_atom,
    make_quantum_system,
    non_bo_hamiltonian,
    optimize_non_bo_step,
    positions_to_relative,
    reduced_mass,
    relative_to_positions,
    sample_non_bo,
)


def _exact_hydrogen_log_psi(system: QuantumSystem) -> jax.Array:
    """Exact analytical non-BO ground state log|psi| = -mu_{eN} * ||r_e - R_N||."""
    mu = reduced_mass(system.masses[0], system.masses[1])
    r = jnp.linalg.norm(system.positions[0] - system.positions[1])
    return -mu * r


def test_exact_non_bo_hydrogen_constant_local_energy():
    """Exact non-BO Hydrogen wavefunction gives constant E_L = -mu_{ep}/2 at all positions."""
    mu_ep = reduced_mass(ELECTRON_MASS, PROTON_MASS)
    expected_energy = -0.5 * mu_ep  # ~ -0.49972784 Ha

    key = jax.random.key(42)
    # Test across 16 arbitrary (non-zero CM, arbitrary separation) configurations
    raw_positions = jax.random.normal(key, shape=(16, 2, 3)) * 1.5
    # Ensure particles aren't coincident
    raw_positions = raw_positions.at[:, 0, :].add(jnp.array([0.7, -0.3, 0.4]))

    wf_ansatz = NonBOWaveFunction(
        num_particles=2,
        use_neural_jastrow=False,
        lock_cusp=True,
        init_alpha_scale=1.0,
    )

    for pos in raw_positions:
        sys = make_hydrogen_atom(positions=pos, isotope="H")

        # 1. Analytical log|psi| with mass-weighted single-pass folx
        e_analytic, parts_analytic = non_bo_hamiltonian(
            _exact_hydrogen_log_psi, sys, kinetic_method="mass_weighted"
        )
        assert jnp.isclose(e_analytic, expected_energy, atol=1e-5), (
            f"Analytical E_L={e_analytic}, expected={expected_energy}"
        )

        # 2. Per-particle folx must match mass-weighted single-pass folx
        e_per_particle, _ = non_bo_hamiltonian(
            _exact_hydrogen_log_psi, sys, kinetic_method="per_particle"
        )
        assert jnp.isclose(e_per_particle, e_analytic, atol=1e-5)

        # 3. Verify exact mass ratio of per-particle kinetic energies: T_p / T_e = m_e / m_p
        t_per = calculate_per_particle_kinetic_energy(_exact_hydrogen_log_psi, sys)
        assert t_per.shape == (2,)
        assert jnp.isclose(t_per[1] / t_per[0], ELECTRON_MASS / PROTON_MASS, rtol=1e-4)

        # 4. NonBOWaveFunction at init_alpha_scale=1.0 reproduces exact Hydrogen ground state
        e_ansatz, _ = non_bo_hamiltonian(wf_ansatz, sys)
        assert jnp.isclose(e_ansatz, expected_energy, atol=1e-5)


def test_isotope_shift_hydrogen_vs_deuterium():
    """Verify exact non-BO isotope shift E_0(D) - E_0(H) = -0.5 * (mu_ed - mu_ep)."""
    mu_ep = reduced_mass(ELECTRON_MASS, PROTON_MASS)
    mu_ed = reduced_mass(ELECTRON_MASS, DEUTERON_MASS)
    expected_h = -0.5 * mu_ep
    expected_d = -0.5 * mu_ed
    expected_shift = expected_d - expected_h  # ~ -1.3607e-4 Ha

    pos = jnp.array([[1.2, -0.5, 0.8], [-0.4, 0.3, -0.2]], dtype=jnp.float32)
    sys_h = make_hydrogen_atom(positions=pos, isotope="H")
    sys_d = make_hydrogen_atom(positions=pos, isotope="D")

    wf = NonBOWaveFunction(num_particles=2, use_neural_jastrow=False, init_alpha_scale=1.0)
    e_h, _ = non_bo_hamiltonian(wf, sys_h)
    e_d, _ = non_bo_hamiltonian(wf, sys_d)

    assert jnp.isclose(e_h, expected_h, atol=1e-5)
    assert jnp.isclose(e_d, expected_d, atol=1e-5)
    assert e_d < e_h  # Heavier isotope has lower ground-state energy
    assert jnp.isclose(e_d - e_h, expected_shift, atol=1e-5)

    # Born-Oppenheimer infinite-mass limit recovers -0.5 Ha
    sys_bo = make_quantum_system(
        positions=pos,
        charges=jnp.array([-1.0, 1.0]),
        masses=jnp.array([1.0, 1e10]),
    )
    e_bo, _ = non_bo_hamiltonian(_exact_hydrogen_log_psi, sys_bo)
    assert jnp.isclose(e_bo, -0.5, atol=1e-5)


def test_harmonic_cm_confinement_exact_separation():
    """Harmonic CM envelope + V_CM adds exact +1.5 * omega_CM to the internal energy."""
    cm_omega = 0.4
    mu_ep = reduced_mass(ELECTRON_MASS, PROTON_MASS)
    expected_total = -0.5 * mu_ep + 1.5 * cm_omega

    wf = NonBOWaveFunction(
        num_particles=2,
        use_neural_jastrow=False,
        init_alpha_scale=1.0,
        cm_omega=cm_omega,
    )
    # Use positions with a non-zero center of mass
    pos = jnp.array([[1.8, -0.4, 0.9], [0.5, 0.2, -0.3]], dtype=jnp.float32)
    sys = make_hydrogen_atom(positions=pos, isotope="H")
    assert jnp.linalg.norm(center_of_mass(sys.positions, sys.masses)) > 0.1

    e_tot, parts = non_bo_hamiltonian(wf, sys, cm_omega=cm_omega)
    assert parts.cm_potential > 0.0
    assert jnp.isclose(e_tot, expected_total, atol=1e-4)


def test_relative_coordinate_roundtrip_and_zero_cm():
    """Relative coordinates reconstruct exact zero-CM positions preserving pairwise distances."""
    sys = make_h2_plus(
        positions=jnp.array(
            [[1.2, -0.3, 0.5], [0.4, 0.8, -1.1], [0.4, 0.8, 0.9]], dtype=jnp.float32
        )
    )
    rel = positions_to_relative(sys.positions)
    assert rel.shape == (2, 3)

    reconstructed = relative_to_positions(rel, sys.masses)
    assert reconstructed.shape == (3, 3)
    assert jnp.allclose(center_of_mass(reconstructed, sys.masses), 0.0, atol=1e-5)

    # All pairwise differences must be identical
    orig_diffs = sys.positions[:, None, :] - sys.positions[None, :, :]
    rec_diffs = reconstructed[:, None, :] - reconstructed[None, :, :]
    assert jnp.allclose(orig_diffs, rec_diffs, atol=1e-5)


def test_non_bo_wavefunction_invariances_and_cusp():
    """NonBOWaveFunction is translation/rotation invariant and preserves exact Kato cusps."""
    wf = NonBOWaveFunction(
        num_particles=3,
        hidden_dim=16,
        use_neural_jastrow=True,
        lock_cusp=True,
        rngs=nnx.Rngs(7),
    )
    # Give final layer non-zero weights so the neural Jastrow is actively contributing
    wf.fc3.kernel.value = jax.random.normal(jax.random.key(99), wf.fc3.kernel.value.shape) * 0.2

    sys = make_h2_plus()
    val0 = wf(sys)

    # 1. Global translation invariance
    shift = jnp.array([12.3, -4.5, 7.8], dtype=jnp.float32)
    sys_shifted = make_h2_plus(positions=sys.positions + shift[None, :])
    assert jnp.isclose(wf(sys_shifted), val0, atol=1e-5)

    # 2. Global 3D rotation invariance (90-degree rotation around z-axis)
    rot = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)
    sys_rot = make_h2_plus(positions=sys.positions @ rot.T)
    assert jnp.isclose(wf(sys_rot), val0, atol=1e-5)

    # 3. Exact Kato cusp slope at small r_{e, p1} even with active neural Jastrow
    mu_ep = reduced_mass(ELECTRON_MASS, PROTON_MASS)
    expected_cusp = -mu_ep  # q_e * q_p * mu_ep = -1 * 1 * mu_ep

    def _log_psi_of_r(r_val: jax.Array) -> jax.Array:
        pos = jnp.array(
            [[r_val, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=jnp.float32
        )
        return wf(make_h2_plus(pos))

    slope = jax.grad(_log_psi_of_r)(jnp.array(1e-3, dtype=jnp.float32))
    assert jnp.isclose(slope, expected_cusp, rtol=1e-2)


def test_vmc_optimization_non_bo_hydrogen():
    """End-to-end VMC on non-BO Hydrogen recovers ground state from a perturbed exponent."""
    mu_ep = reduced_mass(ELECTRON_MASS, PROTON_MASS)
    exact_e0 = -0.5 * mu_ep

    # Start with a significantly perturbed exponent alpha = 0.6 * mu_ep (E ~ -0.42 Ha)
    wf = NonBOWaveFunction(
        num_particles=2,
        use_neural_jastrow=False,
        lock_cusp=False,
        init_alpha_scale=0.6,
    )
    optimizer = nnx.Optimizer(wf, optax.adam(5e-2), wrt=nnx.Param)
    sys_template = make_hydrogen_atom(isotope="H")

    key = jax.random.key(0)
    walkers = None
    energies = []

    for step in range(25):
        key, sample_key = jax.random.split(key)
        samples, acc_rate, walkers = sample_non_bo(
            wf,
            sys_template,
            sample_key,
            num_chains=32,
            num_samples=80,
            burnin=20,
            thinning=2,
            step_size=0.8,
            sampler="rmh",
            mode="relative",
            init_positions=walkers,
        )
        assert 0.15 < float(acc_rate) < 0.95
        mean_e, _std_e, _parts = optimize_non_bo_step(
            wf, optimizer, samples, sys_template
        )
        energies.append(float(mean_e))

    initial_energy = energies[0]
    final_energy = float(jnp.mean(jnp.array(energies[-5:])))
    learned_alpha = float(mu_ep * jnp.exp(wf.log_alpha.value[0]))

    # Initial energy should be well above exact_e0 (~ -0.42 Ha)
    assert initial_energy > exact_e0 + 0.03
    # Final energy should converge closely to exact_e0 (~ -0.4997 Ha)
    assert final_energy < initial_energy - 0.04
    assert abs(final_energy - exact_e0) < 0.015
    assert abs(learned_alpha - mu_ep) < 0.08


def test_vmc_non_bo_h2_plus_and_lab_cm_sampling():
    """Verify neural NonBOWaveFunction VMC on 3-body H2+ and lab-frame CM-confined sampling."""
    # 1. 3-body non-BO H2+ (1 electron + 2 quantum protons) in relative coordinates
    sys_h2p = make_h2_plus(isotopes=("H", "H"))
    wf_h2p = NonBOWaveFunction(
        num_particles=3,
        hidden_dim=12,
        use_neural_jastrow=True,
        lock_cusp=True,
        rngs=nnx.Rngs(123),
    )
    opt_h2p = nnx.Optimizer(wf_h2p, optax.adam(1e-2), wrt=nnx.Param)

    key = jax.random.key(1)
    samples, acc_rate, walkers = sample_non_bo(
        wf_h2p,
        sys_h2p,
        key,
        num_chains=16,
        num_samples=40,
        burnin=10,
        thinning=2,
        step_size=0.3,
        sampler="rmh",
        mode="relative",
    )
    assert samples.shape == (16 * 15, 3, 3)
    assert 0.05 < float(acc_rate) < 0.99
    # Verify sampled positions have zero CM
    cms = jax.vmap(lambda p: center_of_mass(p, sys_h2p.masses))(samples)
    assert jnp.allclose(cms, 0.0, atol=1e-5)

    mean_e, std_e, parts = optimize_non_bo_step(wf_h2p, opt_h2p, samples, sys_h2p)
    assert jnp.isfinite(mean_e) and jnp.isfinite(std_e)
    assert jnp.isfinite(parts.kinetic_energy) and jnp.isfinite(parts.potential_energy)

    # 2. Lab-frame MALA sampling with harmonic CM confinement on Hydrogen
    # Note: CM curvature is 2 * M * cm_omega (~ 184 for cm_omega=0.05), so MALA step_size << 1 / (2 M cm_omega)
    cm_omega = 0.05
    sys_h = make_hydrogen_atom(isotope="H")
    wf_cm = NonBOWaveFunction(
        num_particles=2,
        use_neural_jastrow=False,
        init_alpha_scale=1.0,
        cm_omega=cm_omega,
    )
    samples_lab, acc_lab, _ = sample_non_bo(
        wf_cm,
        sys_h,
        jax.random.key(2),
        num_chains=8,
        num_samples=30,
        burnin=10,
        thinning=2,
        step_size=1e-3,
        sampler="mala",
        mode="lab",
        init_sigma=0.05,
    )
    assert samples_lab.shape == (8 * 10, 2, 3)
    assert jnp.all(jnp.isfinite(samples_lab))
    assert float(acc_lab) > 0.1
    e_lab, _ = jax.vmap(lambda p: non_bo_hamiltonian(wf_cm, make_hydrogen_atom(p), cm_omega=cm_omega))(
        samples_lab[:16]
    )
    expected_cm_e = -0.5 * reduced_mass(ELECTRON_MASS, PROTON_MASS) + 1.5 * cm_omega
    assert jnp.allclose(e_lab, expected_cm_e, atol=1e-3)
