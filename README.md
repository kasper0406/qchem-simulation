## qchem-simulation

Neural variational Monte Carlo (VMC) playground for small quantum chemistry systems. It models electronic structure with a Slater-Jastrow-style wave function parameterized by neural networks (Flax nnx), samples electrons via MCMC (BlackJAX), and optionally optimizes nuclear positions.

### Code layout

- `wavefunction.py` contains the Slater–Jastrow model, orbital networks, and geometric features.
- `hamiltonian.py` contains the Coulomb potentials and forward-Laplacian kinetic energy.
- `sampling.py` contains electron initialization, MALA transitions, and sample collection.
- `electronic_optimization.py` combines samples and local energies into electronic VMC updates and re-exports the existing electronic API.
- `nuclei_optimization.py` estimates nuclear gradients and advances the geometry.
