from absl import app
import einops
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import os
from jaxtyping import Array, Key
import plotly.graph_objects as go

from .electronic_optimization import WaveFunction, train_wavefunction, sample_from_wavefunction
from .nuclei_optimization import optimize_nuclei
from .utils import Nucleus

from folx import register_function, wrap_forward_laplacian


def jiggle_nuclei(nuclei_positions: Array, key: Key, scale: float = 1e-2):
    noise = jax.random.normal(key, nuclei_positions.shape) * scale
    return nuclei_positions + noise


def plot_nuclei_and_electrons(
    step: int,
    wave_function: WaveFunction,
    num_electron_chains: int,
    num_electron_samples: int,
    nuclei: Nucleus,
    key: Key
):
    sum_of_charges = int(jnp.sum(nuclei.charge))
    electron_samples, _log_densities, _acceptance_rate = sample_from_wavefunction(
        wave_function=wave_function,
        nuclei=nuclei,
        sum_of_charges=sum_of_charges,
        key=key,
        num_chains=num_electron_chains,
        num_samples=num_electron_samples,
    )
    electron_samples = jax.tree.map(lambda x: einops.rearrange(x, "n c s ... -> (n c s) ..."), electron_samples)

    electron_trace = go.Scatter3d(
        x=electron_samples.position[:, 0],
        y=electron_samples.position[:, 1],
        z=electron_samples.position[:, 2],
        mode='markers',
        name='Electrons',
        marker=dict(size=2, color='rgba(0,120,255,0.7)'),
        hoverinfo='skip',
    )

    nuclei_trace = go.Scatter3d(
        x=nuclei.position[:, 0],
        y=nuclei.position[:, 1],
        z=nuclei.position[:, 2],
        mode='markers',
        name='Nuclei',
        marker=dict(size=6, color='crimson', symbol='diamond'),
        customdata=nuclei.charge,
        hovertemplate='Nucleus<br>Charge Z=%{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>',
    )

    fig = go.Figure(data=[electron_trace, nuclei_trace])
    fig.update_layout(
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='data'),
        legend=dict(title='Molecule')
    )
    # fig.show(renderer="jupyterlab")
    fig.write_html(f"electron_samples/step_{step}.html")


def register_special_folx_functions():
    # Register special forward laplacian (folx) functions
    register_function('glu', wrap_forward_laplacian(jax.nn.glu, in_axes=()))


def main(_args):
    register_special_folx_functions()

    main_key = jax.random.key(0)

    optimization_key, jiggle_key = jax.random.split(main_key, 2)

    ### 1 Oxygen and 2 Hydrogen nuclei (H_2O)
    nuclei_positions = jnp.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, -0.5]], dtype=jnp.float32)
    nuclei_charges = jnp.array([8, 1, 1], dtype=jnp.float32)
    nuclei = Nucleus(position=jiggle_nuclei(nuclei_positions, jiggle_key), charge=nuclei_charges)

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
    electronic_optimizer = nnx.ModelAndOptimizer(
        wave_function,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(1e-3),
        )
    )

    nuclei_optimizer = optax.adam(5e-2)
    # nuclei_optimizer = optax.sgd(0.1)
    nuclei_opt_state = nuclei_optimizer.init(nuclei.position)

    iterations = 100
    iteration_keys = jax.random.split(optimization_key, iterations)
    for iteration, key in enumerate(iteration_keys):
        print(f"Iteration {iteration + 1}/{iterations}")
        electronic_key, nuclei_key, plot_key = jax.random.split(key, num=3)

        electronic_steps = 25 # 100
        nuclei_steps = 1
        num_chains = 50
        num_electronic_samples = 250
        num_nuclei_samples = 650

        train_wavefunction(
            wave_function,
            electronic_optimizer,
            nuclei,
            num_steps=electronic_steps,
            num_samples=num_electronic_samples,
            num_chains=num_chains,
            key=electronic_key
        )

        nuclei_before = nuclei
        nuclei, nuclei_opt_state = optimize_nuclei(
            wave_function,
            nuclei_optimizer,
            nuclei_opt_state,
            nuclei_steps,
            nuclei,
            num_electron_samples=num_nuclei_samples,
            num_electron_chains=num_chains,
            key=nuclei_key,
        )

        plot_nuclei_and_electrons(
            step=iteration,
            wave_function=wave_function,
            num_electron_chains=num_chains,
            num_electron_samples=num_nuclei_samples,
            nuclei=nuclei_before,
            key=plot_key,
        )


if __name__ == "__main__":
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_disable_jit", True)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.90'
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_gemm=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
        '--xla_gpu_all_reduce_combine_threshold_bytes=51200 '
        '--xla_gpu_graph_level=0 '
        # '--xla_gpu_autotune_level=1 '
        '--xla_gpu_per_fusion_autotune_cache_dir=xla_autotune_results '

        '--xla_gpu_strict_conv_algorithm_picker=false '

        # '--xla_dump_to="/home/knielsen/xla_cuda_crash" '
    )

    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    jax.threefry_partitionable(True)

    app.run(main)
