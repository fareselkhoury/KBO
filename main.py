import jax
jax.config.update("jax_enable_x64", True)
import jax.random as random
import jax.numpy as jnp
import os
import argparse

from iv_regression import IVRegression
from kernel import Kernel
from generate_synthetic_data import sample_data
from utils import save_if_not_exists, compute_value_gradient_truth
from config import p, D, d, N, sigma, epsilon, n_sim, lam, n_values

def main(block_dir, traj_dir):
    M=N
    omega_star = random.uniform(random.PRNGKey(0), shape=(d,))
    os.makedirs(traj_dir, exist_ok=True)

    A_file = os.path.join(block_dir, f"A_p{p}_d{d}_N{N}_M{M}_blocksize1000_D{D}_sigma{sigma}_lambda{lam}.npy")
    B_file = os.path.join(block_dir, f"B_p{p}_d{d}_N{N}_M{M}_blocksize1000_D{D}_sigma{sigma}_lambda{lam}.npy")
    yTilde_norm_squared_file = os.path.join(block_dir, f"yTilde_norm_squared_p{p}_d{d}_N{N}_M{M}_blocksize1000_D{D}_sigma{sigma}_lambda{lam}.npy")

    A = jnp.load(A_file)
    B = jnp.load(B_file)
    yTilde_norm_squared = jnp.load(yTilde_norm_squared_file)

    for n in n_values:
        m = n
        for l in range(1, n_sim + 1):
            key = random.PRNGKey(l)
            key_kernel, key_sim = random.split(key, 2)

            x, x_tilde, t, y_tilde = sample_data(key_sim, p, d, n, m, omega_star)
            kernel = Kernel(D, sigma, key_kernel, False)
            model = IVRegression(x, t, x_tilde, y_tilde, lam, d, kernel, False)

            omega_initial = random.uniform(key_sim, shape=(d,))
            omega_trajectory, value_plugin_trajectory, gradient_plugin_trajectory = model.optimize(omega_initial, epsilon)

            vectorized_fn = jax.vmap(compute_value_gradient_truth, in_axes=(0, None, None, None, None))

            value_truth_trajectory, gradient_truth_trajectory = vectorized_fn(jnp.array(omega_trajectory), A, B, yTilde_norm_squared, M)

            omega_file_name = f"omega_trajectory_n{n}_sim{l}_lam{lam}.npy"
            grad_truth_file_name = f"grad_truth_trajectory_n{n}_sim{l}_lam{lam}.npy"
            value_truth_file_name = f"value_truth_trajectory_n{n}_sim{l}_lam{lam}.npy"
            grad_plugin_file_name = f"grad_plugin_trajectory_n{n}_sim{l}_lam{lam}.npy"
            value_plugin_file_name = f"value_plugin_trajectory_n{n}_sim{l}_lam{lam}.npy"

            omega_path = os.path.join(traj_dir, omega_file_name)
            grad_truth_path = os.path.join(traj_dir, grad_truth_file_name)
            value_truth_path = os.path.join(traj_dir, value_truth_file_name)
            grad_plugin_path = os.path.join(traj_dir, grad_plugin_file_name)
            value_plugin_path = os.path.join(traj_dir, value_plugin_file_name)

            save_if_not_exists(omega_path, jnp.array(omega_trajectory))
            save_if_not_exists(grad_truth_path, jnp.array(gradient_truth_trajectory))
            save_if_not_exists(value_truth_path, jnp.array(value_truth_trajectory))
            save_if_not_exists(grad_plugin_path, jnp.array(gradient_plugin_trajectory))
            save_if_not_exists(value_plugin_path, jnp.array(value_plugin_trajectory))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_dir", type=str, required=True, help="Directory where the blocks are")
    parser.add_argument("--traj_dir", type=str, required=True, help="Directory where the trajectories will be saved")
    args = parser.parse_args()
    block_dir = args.block_dir
    traj_dir = args.traj_dir
    main(block_dir, traj_dir)
