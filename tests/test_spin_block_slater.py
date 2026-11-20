import chex
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qchem_simulation.electronic_optimization import (
    DistanceEncoder,
    DistanceEncoders,
    FlatDistanceEncoder,
    OrbitalNetwork,
    SlaterDeterminant,
    WaveFunction,
    calculate_distances,
    calculate_kinetic_energy,
    init_electrons,
)
from qchem_simulation.utils import Electron, Nucleus


def _create_distance_encoders(hdim: int, rngs: nnx.Rngs) -> DistanceEncoders:
    return DistanceEncoders(
        electron=DistanceEncoder(hdim, buckets=16, rngs=rngs),
        electron_nuclei=DistanceEncoder(hdim, buckets=16, rngs=rngs),
        nuclei=FlatDistanceEncoder(hdim, buckets=16, rngs=rngs),
    )


def _alternating_spins(num_electrons: int) -> jax.Array:
    return jnp.reshape((-1.0) ** jnp.arange(num_electrons), (num_electrons, 1))


def test_orbital_network_block_shapes():
    num_electrons = 5
    num_nuclei = 2
    hdim = 8
    rngs = nnx.Rngs(0)

    distance_encoders = _create_distance_encoders(hdim, rngs)
    net = OrbitalNetwork(hdim, num_electrons, num_nuclei, rngs=rngs)
    assert net.num_up == 3
    assert net.num_down == 2

    electron_positions = jax.random.normal(rngs.electron(), shape=(num_electrons, 3))
    nucleus_positions = jax.random.normal(rngs.nucleus(), shape=(num_nuclei, 3))
    dist = calculate_distances(electron_positions, nucleus_positions)
    spins = _alternating_spins(num_electrons)

    (up_orbitals, down_orbitals), ortho = net(dist, spins, distance_encoders=distance_encoders)
    assert up_orbitals is not None and up_orbitals.shape == (3, 3)
    assert down_orbitals is not None and down_orbitals.shape == (2, 2)
    assert ortho.shape == ()
    assert jnp.isfinite(ortho) and ortho >= 0.0


def test_same_spin_swap_flips_sign():
    num_electrons = 5
    num_nuclei = 2
    hdim = 8
    rngs = nnx.Rngs(1)

    distance_encoders = _create_distance_encoders(hdim, rngs)
    slater_det = SlaterDeterminant(num_electrons, num_nuclei, hdim, rngs=rngs)

    electron_positions = jax.random.normal(rngs.electron(), shape=(num_electrons, 3))
    nucleus_positions = jax.random.normal(rngs.nucleus(), shape=(num_nuclei, 3))
    spins = _alternating_spins(num_electrons)  # [+1, -1, +1, -1, +1]

    dist = calculate_distances(electron_positions, nucleus_positions)
    orig_logdet, orig_sign, orig_ortho = slater_det(dist, spins, distance_encoders=distance_encoders)

    # Same-spin pairs: (0, 2), (0, 4), (2, 4) are all spin-up; (1, 3) are both spin-down
    for i, j in [(0, 2), (0, 4), (2, 4), (1, 3)]:
        swapped_positions = electron_positions.at[[i, j], :].set(electron_positions[[j, i], :])
        swapped_dist = calculate_distances(swapped_positions, nucleus_positions)
        swapped_logdet, swapped_sign, swapped_ortho = slater_det(
            swapped_dist, spins, distance_encoders=distance_encoders
        )
        chex.assert_trees_all_close(swapped_logdet, orig_logdet, rtol=1e-4, atol=1e-5)
        chex.assert_trees_all_close(swapped_sign, -orig_sign, rtol=1e-4, atol=1e-5)
        chex.assert_trees_all_close(swapped_ortho, orig_ortho, rtol=1e-4, atol=1e-5)


def test_opposite_spin_label_swap_flips_sign():
    num_electrons = 5
    num_nuclei = 2
    hdim = 8
    rngs = nnx.Rngs(2)

    distance_encoders = _create_distance_encoders(hdim, rngs)
    slater_det = SlaterDeterminant(num_electrons, num_nuclei, hdim, rngs=rngs)

    electron_positions = jax.random.normal(rngs.electron(), shape=(num_electrons, 3))
    nucleus_positions = jax.random.normal(rngs.nucleus(), shape=(num_nuclei, 3))
    spins = _alternating_spins(num_electrons)  # [+1, -1, +1, -1, +1]

    dist = calculate_distances(electron_positions, nucleus_positions)
    orig_logdet, orig_sign, orig_ortho = slater_det(dist, spins, distance_encoders=distance_encoders)

    # Swapping both positions and spins exchanges complete electron labels.
    for i, j in [(0, 1), (0, 3), (1, 2), (1, 4), (3, 4)]:
        swapped_positions = electron_positions.at[[i, j], :].set(electron_positions[[j, i], :])
        swapped_spins = spins.at[[i, j], :].set(spins[[j, i], :])
        swapped_dist = calculate_distances(swapped_positions, nucleus_positions)
        swapped_logdet, swapped_sign, swapped_ortho = slater_det(
            swapped_dist, swapped_spins, distance_encoders=distance_encoders
        )
        chex.assert_trees_all_close(swapped_logdet, orig_logdet, rtol=1e-4, atol=1e-5)
        chex.assert_trees_all_close(swapped_sign, -orig_sign, rtol=1e-4, atol=1e-5)
        chex.assert_trees_all_close(swapped_ortho, orig_ortho, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "num_electrons,num_up,num_nuclei,spins_list",
    [
        (1, 1, 2, [1.0]),  # Single electron (e.g., H2+)
        (2, 1, 2, [1.0, -1.0]),  # Closed-shell 2-electron (e.g., H2)
        (5, 3, 2, [1.0, -1.0, 1.0, -1.0, 1.0]),  # Odd electron count
        (4, 4, 2, [1.0, 1.0, 1.0, 1.0]),  # Fully spin-up polarized
        (4, 0, 2, [-1.0, -1.0, -1.0, -1.0]),  # Fully spin-down polarized
        (10, 5, 3, [1.0, -1.0] * 5),  # H2O (5 up, 5 down, 3 nuclei)
    ],
)
def test_slater_determinant_edge_cases(num_electrons, num_up, num_nuclei, spins_list):
    hdim = 8
    rngs = nnx.Rngs(42)
    distance_encoders = _create_distance_encoders(hdim, rngs)
    slater_det = SlaterDeterminant(num_electrons, num_nuclei, hdim, rngs=rngs, num_up=num_up)

    electron_positions = jax.random.normal(rngs.electron(), shape=(num_electrons, 3))
    nucleus_positions = jax.random.normal(rngs.nucleus(), shape=(num_nuclei, 3))
    spins = jnp.array(spins_list, dtype=jnp.float32)[:, None]

    dist = calculate_distances(electron_positions, nucleus_positions)
    logdet, sign, ortho = slater_det(dist, spins, distance_encoders=distance_encoders)

    assert logdet.shape == ()
    assert sign.shape == ()
    assert ortho.shape == ()
    assert jnp.isfinite(logdet)
    assert jnp.isfinite(ortho) and ortho >= 0.0
    assert float(jnp.abs(sign)) == pytest.approx(1.0, abs=1e-5)

    # Verify antisymmetry for any same-spin pair if at least 2 same-spin electrons exist
    num_down = num_electrons - num_up
    if num_up >= 2:
        up_idx = [idx for idx, s in enumerate(spins_list) if s > 0]
        i, j = up_idx[0], up_idx[1]
        swapped_pos = electron_positions.at[[i, j], :].set(electron_positions[[j, i], :])
        swapped_logdet, swapped_sign, _ = slater_det(
            calculate_distances(swapped_pos, nucleus_positions), spins, distance_encoders=distance_encoders
        )
        chex.assert_trees_all_close(swapped_logdet, logdet, rtol=1e-4, atol=1e-5)
        chex.assert_trees_all_close(swapped_sign, -sign, rtol=1e-4, atol=1e-5)

    if num_down >= 2:
        down_idx = [idx for idx, s in enumerate(spins_list) if s <= 0]
        i, j = down_idx[0], down_idx[1]
        swapped_pos = electron_positions.at[[i, j], :].set(electron_positions[[j, i], :])
        swapped_logdet, swapped_sign, _ = slater_det(
            calculate_distances(swapped_pos, nucleus_positions), spins, distance_encoders=distance_encoders
        )
        chex.assert_trees_all_close(swapped_logdet, logdet, rtol=1e-4, atol=1e-5)
        chex.assert_trees_all_close(swapped_sign, -sign, rtol=1e-4, atol=1e-5)


def test_invalid_num_up_raises():
    rngs = nnx.Rngs(0)
    with pytest.raises(ValueError):
        OrbitalNetwork(8, num_electrons=4, num_nuclei=2, rngs=rngs, num_up=-1)
    with pytest.raises(ValueError):
        OrbitalNetwork(8, num_electrons=4, num_nuclei=2, rngs=rngs, num_up=5)
    with pytest.raises(ValueError):
        WaveFunction(num_electrons=4, num_nuclei=2, rngs=rngs, num_up=5)


@pytest.mark.parametrize(
    "num_electrons,num_up,charges,spins_list",
    [
        (1, 1, [1.0, 1.0], [1.0]),  # H2+ (1 up, 0 down)
        (2, 1, [1.0, 1.0], [1.0, -1.0]),  # H2 (1 up, 1 down)
        (4, 4, [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]),  # Polarized (4 up, 0 down)
        (5, 3, [3.0, 2.0], [1.0, -1.0, 1.0, -1.0, 1.0]),  # Odd (3 up, 2 down)
    ],
)
def test_wavefunction_jit_grad_and_kinetic_energy(num_electrons, num_up, charges, spins_list):
    num_nuclei = len(charges)
    rngs = nnx.Rngs(7)
    wf = WaveFunction(num_electrons=num_electrons, num_nuclei=num_nuclei, rngs=rngs, num_up=num_up)

    nuclei = Nucleus(
        position=jax.random.normal(rngs.nucleus(), shape=(num_nuclei, 3)),
        charge=jnp.array(charges, dtype=jnp.float32),
    )
    electrons = Electron(
        position=jax.random.normal(rngs.electron(), shape=(num_electrons, 3)),
        spin=jnp.array(spins_list, dtype=jnp.float32)[:, None],
    )

    # 1. nnx.jit forward pass
    @nnx.jit
    def eval_wf(model, elec, nuc):
        return model(elec, nuc)

    log_psi, sign, ortho = eval_wf(wf, electrons, nuclei)
    assert jnp.isfinite(log_psi)
    assert float(jnp.abs(sign)) == pytest.approx(1.0, abs=1e-5)
    assert jnp.isfinite(ortho)

    # 2. nnx.grad wrt model parameters
    @nnx.jit
    def loss_and_grad(model, elec, nuc):
        def loss_fn(m):
            lp, _, _ = m(elec, nuc)
            return lp

        return nnx.value_and_grad(loss_fn)(model)

    val, grads = loss_and_grad(wf, electrons, nuclei)
    assert jnp.isfinite(val)
    grad_leaves = jax.tree.leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
    assert any(jnp.any(g != 0.0) for g in grad_leaves)

    # 3. calculate_kinetic_energy (folx.forward_laplacian) under nnx.jit
    @nnx.jit
    def kinetic_fn(model, elec, nuc):
        return calculate_kinetic_energy(model, elec, nuc)

    ke = kinetic_fn(wf, electrons, nuclei)
    assert ke.shape == ()
    assert jnp.isfinite(ke)


def test_wavefunction_default_num_up_matches_init_electrons():
    nuclei = Nucleus(
        position=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=jnp.float32),
        charge=jnp.array([3.0, 2.0], dtype=jnp.float32),
    )
    sum_of_charges = 5
    electrons_batch = init_electrons(nuclei, num_chains=2, rng=jax.random.key(0), sum_of_charges=sum_of_charges)
    electrons = Electron(position=electrons_batch.position[0], spin=electrons_batch.spin[0])

    # Default num_up should match init_electrons' alternating spins (+1, -1, +1, -1, +1) -> 3 up, 2 down
    wf = WaveFunction(num_electrons=sum_of_charges, num_nuclei=2, rngs=nnx.Rngs(0))
    assert wf.num_up == 3
    assert wf.num_down == 2
    assert int(jnp.sum(electrons.spin > 0)) == wf.num_up

    log_psi, sign, ortho = wf(electrons, nuclei)
    assert jnp.isfinite(log_psi)
    assert float(jnp.abs(sign)) == pytest.approx(1.0, abs=1e-5)
    assert jnp.isfinite(ortho)

@pytest.mark.parametrize("num_up", [0, 1, 3, 5])
def test_initializer_sets_requested_population(num_up):
    nuclei = Nucleus(position=jnp.zeros((1, 3)), charge=jnp.array([5.0]))
    electrons = init_electrons(nuclei, 2, jax.random.key(1), 5, num_up=num_up)
    assert jnp.all(jnp.sum(electrons.spin > 0, axis=1) == num_up)


def test_wavefunction_has_zero_amplitude_outside_spin_sector():
    nuclei = Nucleus(position=jnp.zeros((1, 3)), charge=jnp.array([2.0]))
    electrons = init_electrons(nuclei, 1, jax.random.key(1), 2)
    electrons = Electron(position=electrons.position[0], spin=electrons.spin[0])
    wf = WaveFunction(2, 1, nnx.Rngs(0), num_up=2)
    log_psi, sign, _ = nnx.jit(lambda model, e, n: model(e, n))(wf, electrons, nuclei)
    assert jnp.isneginf(log_psi)
    assert sign == 0
