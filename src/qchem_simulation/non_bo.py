"""Non-Born-Oppenheimer (Quantum Nuclei + Electrons) Variational Monte Carlo.

Treats both electrons and atomic nuclei as quantum particles with finite masses
in atomic units (hbar = e = m_e = 4*pi*eps_0 = 1).

Key physics:
- Full non-BO Hamiltonian:
      H = -sum_k (1 / (2 m_k)) nabla_{x_k}^2 + sum_{i < j} (q_i q_j) / r_{ij}
- Exact reduced-mass Kato cusp condition for any distinguishable pair (i, j):
      d(log|psi|)/d(r_{ij}) |_{r_{ij} -> 0} = mu_{ij} * q_i * q_j
  where mu_{ij} = (m_i * m_j) / (m_i + m_j).
- Single-pass mass-weighted forward Laplacian via `folx.forward_laplacian`:
  substituting y_k = sqrt(m_k) * x_k yields
      -0.5 * (Delta_y log|psi| + ||nabla_y log|psi||^2)
      = -sum_k (1 / (2 m_k)) (Delta_{x_k} log|psi| + ||nabla_{x_k} log|psi||^2)
  in a single forward-Laplacian pass over all quantum particles.
- Center-of-mass (CM) zero-mode handling via either exact zero-CM relative
  coordinates (default) or optional harmonic CM confinement (cm_omega > 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import blackjax
import einops
import jax
import jax.numpy as jnp
from flax import nnx
from folx import forward_laplacian
from jaxtyping import Array, Key

# ---------------------------------------------------------------------------
# Physical constants in atomic units (CODATA 2018)
# ---------------------------------------------------------------------------
ELECTRON_MASS: float = 1.0
POSITRON_MASS: float = 1.0
PROTON_MASS: float = 1836.15267343
DEUTERON_MASS: float = 3670.48296788
TRITON_MASS: float = 5496.92153573

ISOTOPE_MASSES: dict[str, float] = {
    "H": PROTON_MASS,
    "D": DEUTERON_MASS,
    "T": TRITON_MASS,
}


def reduced_mass(m1: float | Array, m2: float | Array) -> float | Array:
    """Compute the two-body reduced mass mu = (m1 * m2) / (m1 + m2)."""
    return (m1 * m2) / (m1 + m2)


# ---------------------------------------------------------------------------
# Data structures (JAX pytrees)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParticleSpec:
    """Static specification of a single quantum particle."""

    charge: float
    mass: float
    species: int = 0
    spin: float = 0.0
    name: str = ""


@jax.tree_util.register_dataclass
@dataclass
class QuantumSystem:
    """Pytree representing all quantum particles (electrons + quantum nuclei).

    Attributes:
        positions: Particle coordinates in atomic units (bohr), shape (N, 3).
        charges: Particle electric charges in atomic units, shape (N,).
        masses: Particle masses in atomic units (m_e = 1), shape (N,).
        species: Integer species labels (e.g. 0=electron, 1=proton, 2=deuteron), shape (N,).
        spins: Spin projection labels (e.g. +0.5/-0.5, or 0.0 for spinless/distinguishable), shape (N,).
    """

    positions: Array
    charges: Array
    masses: Array
    species: Array
    spins: Array


@jax.tree_util.register_dataclass
@dataclass
class NonBOHamiltonianParts:
    """Breakdown of non-BO local energy contributions."""

    kinetic_energy: Array
    potential_energy: Array
    cm_potential: Array


def make_quantum_system(
    positions: Array,
    charges: Array,
    masses: Array,
    species: Array | None = None,
    spins: Array | None = None,
) -> QuantumSystem:
    """Construct a validated `QuantumSystem` pytree."""
    positions = jnp.asarray(positions, dtype=jnp.float32)
    charges = jnp.asarray(charges, dtype=jnp.float32)
    masses = jnp.asarray(masses, dtype=jnp.float32)
    num_particles = positions.shape[0]
    if species is None:
        species = jnp.arange(num_particles, dtype=jnp.int32)
    else:
        species = jnp.asarray(species, dtype=jnp.int32)
    if spins is None:
        spins = jnp.zeros((num_particles,), dtype=jnp.float32)
    else:
        spins = jnp.asarray(spins, dtype=jnp.float32)
    return QuantumSystem(
        positions=positions,
        charges=charges,
        masses=masses,
        species=species,
        spins=spins,
    )


def from_particle_specs(specs: list[ParticleSpec], positions: Array) -> QuantumSystem:
    """Build a `QuantumSystem` from a list of `ParticleSpec` and positions."""
    return make_quantum_system(
        positions=positions,
        charges=jnp.array([s.charge for s in specs], dtype=jnp.float32),
        masses=jnp.array([s.mass for s in specs], dtype=jnp.float32),
        species=jnp.array([s.species for s in specs], dtype=jnp.int32),
        spins=jnp.array([s.spin for s in specs], dtype=jnp.float32),
    )


def make_hydrogen_atom(
    positions: Array | None = None,
    isotope: str = "H",
) -> QuantumSystem:
    """Create a non-BO Hydrogen-like atom (1 electron + 1 quantum nucleus).

    Args:
        positions: Optional (2, 3) array [electron_pos, nucleus_pos].
        isotope: One of "H" (protium), "D" (deuterium), or "T" (tritium).
    """
    if isotope not in ISOTOPE_MASSES:
        raise ValueError(f"Unknown isotope {isotope!r}; expected one of {list(ISOTOPE_MASSES)}")
    nuc_mass = ISOTOPE_MASSES[isotope]
    if positions is None:
        positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    nuc_species = {"H": 1, "D": 2, "T": 3}[isotope]
    return make_quantum_system(
        positions=positions,
        charges=jnp.array([-1.0, 1.0], dtype=jnp.float32),
        masses=jnp.array([ELECTRON_MASS, nuc_mass], dtype=jnp.float32),
        species=jnp.array([0, nuc_species], dtype=jnp.int32),
        spins=jnp.array([0.5, 0.5], dtype=jnp.float32),
    )


def make_h2_plus(
    positions: Array | None = None,
    isotopes: tuple[str, str] = ("H", "H"),
) -> QuantumSystem:
    """Create a non-BO H2+ molecular ion (1 electron + 2 quantum nuclei).

    Args:
        positions: Optional (3, 3) array [electron_pos, nucleus1_pos, nucleus2_pos].
        isotopes: Tuple of two isotope labels from {"H", "D", "T"}.
    """
    for iso in isotopes:
        if iso not in ISOTOPE_MASSES:
            raise ValueError(f"Unknown isotope {iso!r}; expected one of {list(ISOTOPE_MASSES)}")
    if positions is None:
        positions = jnp.array(
            [[0.0, 0.0, 0.5], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        )
    species_map = {"H": 1, "D": 2, "T": 3}
    return make_quantum_system(
        positions=positions,
        charges=jnp.array([-1.0, 1.0, 1.0], dtype=jnp.float32),
        masses=jnp.array(
            [ELECTRON_MASS, ISOTOPE_MASSES[isotopes[0]], ISOTOPE_MASSES[isotopes[1]]],
            dtype=jnp.float32,
        ),
        species=jnp.array([0, species_map[isotopes[0]], species_map[isotopes[1]]], dtype=jnp.int32),
        # Opposite nuclear spins for ground-state para-H2+ (spatially symmetric nuclei)
        spins=jnp.array([0.5, 0.5, -0.5], dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Center-of-mass & relative coordinate utilities
# ---------------------------------------------------------------------------
def center_of_mass(positions: Array, masses: Array) -> Array:
    """Compute center-of-mass vector R_CM = sum_k (m_k * x_k) / sum_k m_k."""
    total_mass = jnp.sum(masses)
    return jnp.sum(masses[:, None] * positions, axis=0) / total_mass


def subtract_center_of_mass(positions: Array, masses: Array) -> Array:
    """Shift positions so that center_of_mass(shifted, masses) == 0."""
    return positions - center_of_mass(positions, masses)[None, :]


def positions_to_relative(positions: Array) -> Array:
    """Convert (N, 3) lab positions to (N-1, 3) relative coords u_k = x_k - x_{N-1}."""
    return positions[:-1] - positions[-1:]


def relative_to_positions(relative: Array, masses: Array) -> Array:
    """Invert `positions_to_relative` onto the exact zero-CM hyperplane.

    Given u_k = x_k - x_{N-1} for k = 0, ..., N-2 and sum_k m_k x_k = 0:
        x_{N-1} = -(1 / M) * sum_{k=0}^{N-2} m_k * u_k
        x_k     = u_k + x_{N-1}
    This is a linear bijection with constant Jacobian determinant.
    """
    total_mass = jnp.sum(masses)
    x_last = -jnp.sum(masses[:-1, None] * relative, axis=0, keepdims=True) / total_mass
    x_rest = relative + x_last
    return jnp.concatenate([x_rest, x_last], axis=0)


def pairwise_kato_cusps(
    charges: Array,
    masses: Array,
    species: Array | None = None,
    spins: Array | None = None,
) -> Array:
    """Compute exact reduced-mass Kato cusp coefficients c_{ij} for upper-triangular pairs.

    For distinguishable or opposite-spin pairs:
        c_{ij} = mu_{ij} * q_i * q_j
    For identical same-spin fermions (same species and equal non-zero spin):
        c_{ij} = 0.5 * mu_{ij} * q_i * q_j
    """
    n = charges.shape[0]
    idx_i, idx_j = jnp.triu_indices(n, k=1)
    mu_ij = reduced_mass(masses[idx_i], masses[idx_j])
    c_ij = mu_ij * charges[idx_i] * charges[idx_j]
    if species is not None and spins is not None:
        same_fermion = (
            (species[idx_i] == species[idx_j])
            & (spins[idx_i] == spins[idx_j])
            & (spins[idx_i] != 0.0)
        )
        c_ij = jnp.where(same_fermion, 0.5 * c_ij, c_ij)
    return c_ij


# ---------------------------------------------------------------------------
# Non-BO Kinetic Energy & Hamiltonian
# ---------------------------------------------------------------------------
def _make_log_psi_of_positions(
    wave_function: nnx.Module | Callable[[QuantumSystem], Array],
    system: QuantumSystem,
) -> Callable[[Array], Array]:
    """Wrap `wave_function` (nnx.Module or plain callable) as positions -> scalar log|psi|."""
    if isinstance(wave_function, nnx.Module):
        graphdef, state = nnx.split(wave_function)

        def _fn(positions: Array, *, _graphdef=graphdef, _state=state) -> Array:
            wf = nnx.merge(_graphdef, _state)
            sys = QuantumSystem(
                positions=positions,
                charges=system.charges,
                masses=system.masses,
                species=system.species,
                spins=system.spins,
            )
            return jnp.squeeze(wf(sys))

        return _fn

    def _plain_fn(positions: Array) -> Array:
        sys = QuantumSystem(
            positions=positions,
            charges=system.charges,
            masses=system.masses,
            species=system.species,
            spins=system.spins,
        )
        return jnp.squeeze(wave_function(sys))

    return _plain_fn


def calculate_non_bo_kinetic_energy(
    wave_function: nnx.Module | Callable[[QuantumSystem], Array],
    system: QuantumSystem,
    *,
    method: str = "mass_weighted",
) -> Array:
    """Compute the exact non-BO local kinetic energy across all quantum particles.

    Evaluates:
        T_L = -sum_k (1 / (2 m_k)) * (nabla_{x_k}^2 log|psi| + ||nabla_{x_k} log|psi||^2)

    Args:
        wave_function: Callable or `nnx.Module` mapping `QuantumSystem` -> scalar `log|psi|`.
        system: `QuantumSystem` containing positions, charges, masses, species, spins.
        method:
            - "mass_weighted" (default): Single-pass `folx.forward_laplacian` in
              mass-weighted coordinates y_k = sqrt(m_k) * x_k.
            - "per_particle": Individual `folx.forward_laplacian` per particle k,
              weighted by -1 / (2 m_k) and summed.

    Returns:
        Scalar local kinetic energy T_L (in Hartree).
    """
    if method == "mass_weighted":
        log_psi_pos = _make_log_psi_of_positions(wave_function, system)
        num_particles = system.positions.shape[0]
        sqrt_masses = jnp.sqrt(system.masses)
        y_flat = (system.positions * sqrt_masses[:, None]).reshape((-1,))

        def log_psi_mass_weighted(y: Array) -> Array:
            pos = y.reshape((num_particles, 3)) / sqrt_masses[:, None]
            return log_psi_pos(pos)

        lap_res = forward_laplacian(log_psi_mass_weighted)(y_flat)
        grad_sq = jnp.sum(jnp.square(lap_res.dense_jacobian))
        return -0.5 * (lap_res.laplacian + grad_sq)

    if method == "per_particle":
        per_particle = calculate_per_particle_kinetic_energy(wave_function, system)
        return jnp.sum(per_particle)

    raise ValueError(f"Unknown kinetic_method {method!r}; expected 'mass_weighted' or 'per_particle'")


def calculate_per_particle_kinetic_energy(
    wave_function: nnx.Module | Callable[[QuantumSystem], Array],
    system: QuantumSystem,
) -> Array:
    """Compute per-particle local kinetic energies T_k for k = 0, ..., N-1.

    Returns:
        Array of shape (N,) where entry k is
        T_k = -(1 / (2 m_k)) * (nabla_{x_k}^2 log|psi| + ||nabla_{x_k} log|psi||^2).
    """
    log_psi_pos = _make_log_psi_of_positions(wave_function, system)
    indices = jnp.arange(system.positions.shape[0])

    def _single_particle(pos_k: Array, idx: Array, mass_k: Array) -> Array:
        def _fn(x_k: Array) -> Array:
            updated = system.positions.at[idx].set(x_k)
            return log_psi_pos(updated)

        lap_res = forward_laplacian(_fn)(pos_k)
        grad_sq = jnp.sum(jnp.square(lap_res.dense_jacobian))
        return -0.5 * (lap_res.laplacian + grad_sq) / mass_k

    return jax.vmap(_single_particle, in_axes=(0, 0, 0))(
        system.positions, indices, system.masses
    )


def calculate_coulomb_potential(system: QuantumSystem, eps: float = 1e-12) -> Array:
    """Compute full pairwise Coulomb potential V = sum_{i < j} q_i * q_j / r_{ij}."""
    n = system.positions.shape[0]
    if n < 2:
        return jnp.array(0.0, dtype=system.positions.dtype)
    idx_i, idx_j = jnp.triu_indices(n, k=1)
    diff = system.positions[idx_i] - system.positions[idx_j]
    dist = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1) + eps)
    charge_prod = system.charges[idx_i] * system.charges[idx_j]
    return jnp.sum(charge_prod / dist)


def non_bo_hamiltonian(
    wave_function: nnx.Module | Callable[[QuantumSystem], Array],
    system: QuantumSystem,
    *,
    cm_omega: float = 0.0,
    kinetic_method: str = "mass_weighted",
    eps: float = 1e-12,
) -> tuple[Array, NonBOHamiltonianParts]:
    """Evaluate the full non-BO local energy E_L = (H psi) / psi.

    Args:
        wave_function: Callable or `nnx.Module` mapping `QuantumSystem` -> scalar `log|psi|`.
        system: `QuantumSystem` pytree.
        cm_omega: Optional harmonic CM trap frequency omega_CM >= 0. When > 0, adds
            V_CM = 0.5 * M * omega_CM^2 * ||R_CM||^2 (whose exact ground state adds
            +1.5 * omega_CM to the internal energy).
        kinetic_method: Passed to `calculate_non_bo_kinetic_energy`.
        eps: Small softening epsilon inside sqrt(r^2 + eps).

    Returns:
        Tuple `(local_energy, NonBOHamiltonianParts)`.
    """
    kinetic = calculate_non_bo_kinetic_energy(
        wave_function, system, method=kinetic_method
    )
    potential = calculate_coulomb_potential(system, eps=eps)
    if cm_omega > 0.0:
        total_mass = jnp.sum(system.masses)
        r_cm = center_of_mass(system.positions, system.masses)
        cm_pot = 0.5 * total_mass * (cm_omega ** 2) * jnp.sum(jnp.square(r_cm))
    else:
        cm_pot = jnp.array(0.0, dtype=system.positions.dtype)

    local_energy = kinetic + potential + cm_pot
    parts = NonBOHamiltonianParts(
        kinetic_energy=kinetic,
        potential_energy=potential,
        cm_potential=cm_pot,
    )
    return local_energy, parts


# ---------------------------------------------------------------------------
# Translation-Invariant Non-BO Wavefunction Ansatz
# ---------------------------------------------------------------------------
class NonBOWaveFunction(nnx.Module):
    """Translation- and rotation-invariant non-BO wavefunction ansatz.

    Depends strictly on pairwise inter-particle distances r_{ij} (plus an optional
    harmonic center-of-mass envelope when `cm_omega > 0`).

    Structure:
        log|psi| = sum_{i < j} f_cusp(r_{ij}) + sum_{i < j} g_NN(phi(r_{ij})) + log_cm(R_CM)

    where:
    - Attractive pairs (c_{ij} < 0):
        If `lock_cusp=True`:
            f_attr(r) = -alpha * r + ((c + alpha) * r) / (1 + beta * r)
        which satisfies f'_attr(0) = c_{ij} = mu_{ij} * q_i * q_j EXACTLY for all
        alpha, beta > 0, and decays as -alpha * r at large r.
        If `lock_cusp=False`:
            f_attr(r) = -alpha * r
    - Repulsive pairs (c_{ij} > 0):
            f_rep(r) = (c * r) / (1 + beta * r)
        which satisfies f'_rep(0) = c_{ij} EXACTLY and saturates at large r.
    - Optional neural Jastrow `g_NN` takes bounded quadratic distance features
      r^2 / (1 + r^2) and r^2 / (4 + r^2), whose derivatives vanish identically
      at r = 0 so the exact Kato cusp conditions are strictly preserved.
    """

    def __init__(
        self,
        num_particles: int,
        *,
        hidden_dim: int = 16,
        use_neural_jastrow: bool = True,
        lock_cusp: bool = True,
        init_alpha_scale: float = 1.0,
        cm_omega: float = 0.0,
        eps: float = 1e-12,
        rngs: nnx.Rngs | None = None,
    ):
        if num_particles < 2:
            raise ValueError("NonBOWaveFunction requires at least 2 particles.")
        self.num_particles = num_particles
        self.num_pairs = num_particles * (num_particles - 1) // 2
        self.use_neural_jastrow = use_neural_jastrow
        self.lock_cusp = lock_cusp
        self.cm_omega = float(cm_omega)
        self.eps = float(eps)

        self.log_alpha = nnx.Param(
            jnp.full((self.num_pairs,), jnp.log(init_alpha_scale), dtype=jnp.float32)
        )
        self.log_beta = nnx.Param(jnp.zeros((self.num_pairs,), dtype=jnp.float32))

        if use_neural_jastrow:
            if rngs is None:
                rngs = nnx.Rngs(0)
            self.fc1 = nnx.Linear(4, hidden_dim, rngs=rngs)
            self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
            self.fc3 = nnx.Linear(
                hidden_dim,
                1,
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

    def __call__(self, system: QuantumSystem) -> Array:
        idx_i, idx_j = jnp.triu_indices(self.num_particles, k=1)
        diff = system.positions[idx_i] - system.positions[idx_j]
        r_sq = jnp.sum(jnp.square(diff), axis=-1)
        r = jnp.sqrt(r_sq + self.eps)

        c_ij = pairwise_kato_cusps(
            system.charges, system.masses, system.species, system.spins
        )
        alpha = jnp.abs(c_ij) * jnp.exp(self.log_alpha.value)
        beta_attr = jnp.exp(self.log_beta.value)
        # Scale repulsive crossover beta with |c_ij| so heavy-nucleus cusps act on their
        # physical correlation length ~1/mu_{ij} while keeping log|psi| O(1) at bonding distances.
        beta_rep = jnp.maximum(1.0, jnp.abs(c_ij)) * jnp.exp(self.log_beta.value)

        if self.lock_cusp:
            attr_term = -alpha * r + ((c_ij + alpha) * r) / (1.0 + beta_attr * r)
        else:
            attr_term = -alpha * r

        rep_term = (c_ij * r) / (1.0 + beta_rep * r)
        pair_cusp = jnp.where(c_ij < 0.0, attr_term, jnp.where(c_ij > 0.0, rep_term, 0.0))
        log_psi = jnp.sum(pair_cusp)

        if self.use_neural_jastrow:
            # Cusp-safe smooth features: df/dr = 0 at r = 0, bounded as r -> inf
            feat_short = r_sq / (1.0 + r_sq)
            feat_long = r_sq / (4.0 + r_sq)
            q_prod = system.charges[idx_i] * system.charges[idx_j]
            mu_ij = reduced_mass(system.masses[idx_i], system.masses[idx_j])
            log_mu = jnp.log1p(mu_ij)
            feats = jnp.stack([feat_short, feat_long, q_prod, log_mu], axis=-1)

            h = jnp.tanh(self.fc1(feats))
            h = jnp.tanh(self.fc2(h))
            nn_corr = self.fc3(h).squeeze(-1)
            log_psi = log_psi + jnp.sum(nn_corr)

        if self.cm_omega > 0.0:
            total_mass = jnp.sum(system.masses)
            r_cm = center_of_mass(system.positions, system.masses)
            log_psi = log_psi - 0.5 * total_mass * self.cm_omega * jnp.sum(jnp.square(r_cm))

        return jnp.squeeze(log_psi)


# ---------------------------------------------------------------------------
# Joint MCMC Sampling (BlackJAX)
# ---------------------------------------------------------------------------
def sample_non_bo(
    wave_function: nnx.Module | Callable[[QuantumSystem], Array],
    system_template: QuantumSystem,
    key: Key,
    *,
    num_chains: int = 32,
    num_samples: int = 200,
    burnin: int = 50,
    thinning: int = 2,
    step_size: float = 0.5,
    sampler: str = "rmh",
    mode: str = "relative",
    init_positions: Array | None = None,
    init_sigma: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Sample all quantum particle coordinates jointly from |psi|^2 via BlackJAX.

    Args:
        wave_function: `nnx.Module` or callable `QuantumSystem -> log|psi|`.
        system_template: `QuantumSystem` supplying charges, masses, species, spins
            (and default reference positions if `init_positions` is None).
        key: JAX PRNG key.
        num_chains: Number of parallel Markov chains.
        num_samples: Total MCMC steps per chain (before burn-in & thinning).
        burnin: Number of initial steps to discard per chain.
        thinning: Keep every `thinning`-th post-burn-in sample.
        step_size: Proposal scale (std dev for RMH, step size for MALA).
        sampler: "rmh" (Gaussian random-walk MH) or "mala" (Metropolis-adjusted Langevin).
        mode:
            - "relative" (default): Sample the (N-1, 3) relative coordinates and
              reconstruct exact zero-CM lab positions (eliminates CM zero-mode drift).
            - "lab": Sample all (N, 3) lab positions directly (use with `cm_omega > 0`).
        init_positions: Optional (num_chains, N, 3) starting positions (e.g. persistent walkers).
        init_sigma: Gaussian jitter std dev when generating initial positions.

    Returns:
        Tuple `(samples, mean_acceptance_rate, final_positions)`:
        - `samples`: shape `(num_kept * num_chains, N, 3)`.
        - `mean_acceptance_rate`: scalar average acceptance probability.
        - `final_positions`: shape `(num_chains, N, 3)` for persistent walker continuation.
    """
    if mode not in ("relative", "lab"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'relative' or 'lab'")
    if sampler not in ("rmh", "mala"):
        raise ValueError(f"Unknown sampler {sampler!r}; expected 'rmh' or 'mala'")

    init_key, run_key = jax.random.split(key)
    num_particles = system_template.positions.shape[0]
    masses = system_template.masses

    if init_positions is None:
        noise = init_sigma * jax.random.normal(
            init_key, shape=(num_chains, num_particles, 3)
        )
        init_positions = system_template.positions[None, :, :] + noise

    log_psi_pos = _make_log_psi_of_positions(wave_function, system_template)

    if mode == "relative":
        init_coords = jax.vmap(positions_to_relative)(init_positions)

        def logdensity_fn(u: Array) -> Array:
            pos = relative_to_positions(u, masses)
            return 2.0 * log_psi_pos(pos)

    else:
        init_coords = init_positions

        def logdensity_fn(pos: Array) -> Array:
            return 2.0 * log_psi_pos(pos)

    if sampler == "rmh":
        alg = blackjax.additive_step_random_walk.normal_random_walk(
            logdensity_fn, jnp.asarray(step_size, dtype=jnp.float32)
        )
    else:
        alg = blackjax.mala(logdensity_fn, step_size=step_size)

    def _run_chain(chain_init: Array, chain_key: Key):
        state = alg.init(chain_init)

        def _step(carry, step_key):
            next_state, info = alg.step(step_key, carry)
            return next_state, (next_state.position, info.acceptance_rate)

        step_keys = jax.random.split(chain_key, num_samples)
        final_state, (traj, acc_rates) = jax.lax.scan(_step, state, step_keys)
        return traj, acc_rates, final_state.position

    chain_keys = jax.random.split(run_key, num_chains)
    trajs, acc_rates, final_coords = jax.vmap(_run_chain)(init_coords, chain_keys)

    # Apply burn-in and thinning along time axis (axis=1)
    kept_coords = trajs[:, burnin::thinning, ...]
    flat_coords = einops.rearrange(kept_coords, "c t ... -> (c t) ...")

    if mode == "relative":
        samples = jax.vmap(partial(relative_to_positions, masses=masses))(flat_coords)
        final_positions = jax.vmap(partial(relative_to_positions, masses=masses))(final_coords)
    else:
        samples = flat_coords
        final_positions = final_coords

    return samples, jnp.mean(acc_rates), final_positions


# ---------------------------------------------------------------------------
# VMC Optimization Step (Optax + Flax NNX)
# ---------------------------------------------------------------------------
@nnx.jit(static_argnames=("cm_omega", "clip_local_energy_scale", "kinetic_method"))
def optimize_non_bo_step(
    wave_function: NonBOWaveFunction,
    optimizer: nnx.Optimizer,
    samples: Array,
    system_template: QuantumSystem,
    *,
    cm_omega: float = 0.0,
    clip_local_energy_scale: float | None = 5.0,
    kinetic_method: str = "mass_weighted",
) -> tuple[Array, Array, NonBOHamiltonianParts]:
    """Perform one VMC parameter update step on a batch of sampled particle configurations.

    Uses the unbiased score-function (REINFORCE) energy gradient estimator:
        nabla_theta <E> = 2 * E[(E_L - <E_L>) * nabla_theta log|psi_theta|]

    Args:
        wave_function: Trainable `NonBOWaveFunction`.
        optimizer: `nnx.Optimizer` wrapping `wave_function` and an Optax transform.
        samples: Sampled particle positions of shape `(batch_size, N, 3)`.
        system_template: `QuantumSystem` supplying charges, masses, species, spins.
        cm_omega: Optional harmonic CM trap frequency matching the wavefunction.
        clip_local_energy_scale: If not None, clips local energies used in the
            gradient estimator to `median +/- scale * MAD` for outlier robustness.
        kinetic_method: Passed to `non_bo_hamiltonian`.

    Returns:
        Tuple `(mean_energy, energy_std, mean_parts)`.
    """

    def _log_psi_fn(wf: NonBOWaveFunction, positions: Array) -> Array:
        sys = QuantumSystem(
            positions=positions,
            charges=system_template.charges,
            masses=system_template.masses,
            species=system_template.species,
            spins=system_template.spins,
        )
        return wf(sys)

    def _per_sample(wf: NonBOWaveFunction, positions: Array):
        sys = QuantumSystem(
            positions=positions,
            charges=system_template.charges,
            masses=system_template.masses,
            species=system_template.species,
            spins=system_template.spins,
        )
        e_loc, parts = non_bo_hamiltonian(
            wf, sys, cm_omega=cm_omega, kinetic_method=kinetic_method
        )
        grad_log_psi = nnx.grad(_log_psi_fn)(wf, positions)
        return e_loc, parts, grad_log_psi

    local_energies, parts, grads_log_psi = nnx.vmap(
        _per_sample, in_axes=(None, 0)
    )(wave_function, samples)

    mean_energy = jnp.mean(local_energies)
    energy_std = jnp.std(local_energies)

    if clip_local_energy_scale is not None:
        med = jnp.median(local_energies)
        mad = jnp.mean(jnp.abs(local_energies - med)) + 1e-8
        e_for_grad = jnp.clip(
            local_energies,
            med - clip_local_energy_scale * mad,
            med + clip_local_energy_scale * mad,
        )
    else:
        e_for_grad = local_energies

    centered = jax.lax.stop_gradient(e_for_grad - jnp.mean(e_for_grad))

    def _weighted_grad(g: Array) -> Array:
        w = centered.reshape((centered.shape[0],) + (1,) * (g.ndim - 1))
        return 2.0 * jnp.mean(w * g, axis=0)

    param_grads = jax.tree.map(_weighted_grad, grads_log_psi)
    optimizer.update(wave_function, param_grads)

    mean_parts = jax.tree.map(lambda x: jnp.mean(x, axis=0), parts)
    return mean_energy, energy_std, mean_parts
