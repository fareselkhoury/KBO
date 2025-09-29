import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cholesky
import jax.random as random
import argparse

from kernel import Kernel
from generate_synthetic_data import sample_data_blocks
from utils import feature_matrix, save_if_not_exists
from config import D, N, d, p, block_size, sigma, lam

def generate_save_blocks(key, p, d, N, M, omega, block_size, D, sigma, lam, matrices_dir):
    os.makedirs(matrices_dir, exist_ok=True)
    XiTrans_Xi = jnp.zeros((D, D))
    XiTildeTrans_XiTilde = jnp.zeros((D, D))
    XiTrans_F = jnp.zeros((D, d))
    XiTildeTrans_yTilde = jnp.zeros(D)
    yTilde_norm_squared = 0
    block_num = 0
    key_kernel, key_data = random.split(key, 2)

    for x, x_tilde, t, y_tilde in sample_data_blocks(key_data, p, d, N, M, omega, block_size):
        block_num += 1
        print(block_num)
        kernel = Kernel(D, sigma, key_kernel, True)
        Xi, Xi_tilde = kernel.Xi_Xitilde(x, x_tilde)

        XiTrans_Xi += Xi.T @ Xi
        XiTildeTrans_XiTilde += Xi_tilde.T @ Xi_tilde
        XiTrans_F += Xi.T @ feature_matrix(t, d)
        XiTildeTrans_yTilde += Xi_tilde.T @ y_tilde
        yTilde_norm_squared += (jnp.linalg.norm(y_tilde) ** 2)
    
    reg_term = N * lam * jnp.eye(D)
    XiTrans_Xi_reg = XiTrans_Xi + reg_term
    L = cholesky(XiTrans_Xi_reg, lower=True)
    J = cho_solve((L, True), XiTrans_F)

    A = (1 / M) * J.T @ XiTildeTrans_XiTilde @ J
    B = (1 / M) * J.T @ XiTildeTrans_yTilde

    yTilde_norm_squared_file_name = f"yTilde_norm_squared_p{p}_d{d}_N{N}_M{M}_blocksize{block_size}_D{D}_sigma{sigma}_lambda{lam}.npy"
    A_file_name = f"A_p{p}_d{d}_N{N}_M{M}_blocksize{block_size}_D{D}_sigma{sigma}_lambda{lam}.npy"
    B_file_name = f"B_p{p}_d{d}_N{N}_M{M}_blocksize{block_size}_D{D}_sigma{sigma}_lambda{lam}.npy"

    yTilde_norm_squared_path = os.path.join(matrices_dir, yTilde_norm_squared_file_name)
    A_path = os.path.join(matrices_dir, A_file_name)
    B_path = os.path.join(matrices_dir, B_file_name)

    save_if_not_exists(yTilde_norm_squared_path, jnp.array(yTilde_norm_squared))
    save_if_not_exists(A_path, A)
    save_if_not_exists(B_path, B)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory where the blocks will be saved")
    arg = parser.parse_args()
    matrices_dir = arg.dir
    key = random.PRNGKey(0)
    M = N
    omega_star = random.uniform(key, shape=(d,))
    generate_save_blocks(key, p, d, N, M, omega_star, block_size, D, sigma, lam, matrices_dir)
