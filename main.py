from absl import app
import jax
import jax.numpy as jnp
from utils import Nucleus
from flax import nnx
from electronic_optimization import WaveFunction, train_wavefunction
import optax
from nuclei_optimization import optimize_nuclei

def main(_args):
    # 1 Lithium and 1 Hydrogen nuclei
    nuclei_positions = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)
    nuclei_charges = jnp.array([3, 1], dtype=jnp.int32)
    nuclei = Nucleus(position=nuclei_positions, charge=nuclei_charges)

    # Initialize the wave function
    num_electrons = int(jnp.sum(nuclei.charge))
    num_nuclei = int(nuclei.position.shape[0])
    print(f"Number of electrons: {num_electrons}, Number of nuclei: {num_nuclei}")

    wave_function = WaveFunction(
        num_electrons=num_electrons,
        num_nuclei=num_nuclei,
        rngs=nnx.Rngs(0),
    )

    # Start the wave function optimization
    electronic_optimizer = nnx.Optimizer(
        wave_function,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(1e-3),
        )
    )

    nuclei_optimizer = optax.adam(1e-1)
    nuclei_opt_state = nuclei_optimizer.init(nuclei.position)

    iterations = 100

    main_key = jax.random.key(0)
    iteration_keys = jax.random.split(main_key, iterations)
    for iteration, key in enumerate(iteration_keys):
        print(f"Iteration {iteration + 1}/{iterations}")
        electronic_key, nuclei_key = jax.random.split(key)

        electronic_steps = 50
        nuclei_steps = 5
        num_chains = 150

        train_wavefunction(
            wave_function,
            electronic_optimizer,
            nuclei,
            num_steps=electronic_steps,
            num_samples=1000,
            num_chains=num_chains,
            key=electronic_key
        )

        nuclei, nuclei_opt_state = optimize_nuclei(
            wave_function,
            nuclei_optimizer,
            nuclei_opt_state,
            nuclei_steps,
            nuclei,
            key=nuclei_key,
        )


if __name__ == "__main__":
    app.run(main)
