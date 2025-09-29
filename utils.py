import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import os

def sample_gaussian(key, n, p, variance=5.0):
    std_dev = jnp.sqrt(variance)
    return jnp.squeeze(std_dev * random.normal(key, shape=(n, p)))

def sample_student(key, n, p, df=2.5):
    return jnp.squeeze(random.t(key, df=df, shape=(n, p)))

def gd_update(omega, eta, grad):
    new_omega = omega - eta * grad
    return new_omega

def backtracking_line_search(f, gradf, x, eta=1.0, tau=0.5, c=1e-4):
    while f(x - eta * gradf(x)) > f(x) - c * eta * jnp.linalg.norm(gradf(x)) ** 2:
        eta *= tau
    return eta

def save_if_not_exists(file_path, array):
    if not os.path.exists(file_path):
        jnp.save(file_path, array)
    else:
        print(f"File {file_path} already exists. Skipping save.")

def feature_matrix(t, d):
    return jnp.sin(t[:, None] + jnp.arange(1, d + 1))

def gaussian_kernel(x, y, sigma=1.0):
    return jnp.exp(-jnp.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def gaussian_kernel_gram_matrix(X, Y, sigma=1.0):
    return vmap(lambda x: vmap(lambda y: gaussian_kernel(x, y, sigma))(Y))(X)
    
def compute_value_gradient_truth(omega, A, B, yTilde_norm_squared, M):
    value_truth = (1 / 2) * omega.T @ A @ omega - omega.T @ B + (1 / (2 * M)) * (yTilde_norm_squared)
    gradient_truth = A @ omega - B
    return value_truth, gradient_truth
