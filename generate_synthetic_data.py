import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
from utils import sample_gaussian, sample_student
from config import dist, df

def sample_data(key, p, d, n, m, omega):
    key_x, key_eta = random.split(key, 2)
    total_samples = n + m
    if dist == "gaussian":
    	x = sample_gaussian(key_x, total_samples, p, variance=1.0)
    elif dist == "student":
    	x = sample_student(key_x, total_samples, p, df)
    eta = jnp.sqrt(0.1) * random.normal(key_eta, shape=(total_samples,))
    t = 2 * jnp.sum(x, axis=1) + eta
    epsilon = 0.5 * eta
    phi_t = jnp.sin(jnp.expand_dims(t, axis=1) + jnp.arange(1, d+1))
    y = phi_t @ omega + epsilon
    x1, x2 = x[:n], x[n:]
    t1 = t[:n]
    y2 = y[n:]
    return x1, x2, t1, y2

def sample_data_blocks(key, p, d, N, M, omega, block_size=1000):
    key_x, key_eta = random.split(key, 2)
    total_samples = N + M
    ratio_N = block_size * N // total_samples

    for i in range(0, total_samples, block_size):
        key_x, subkey_x = random.split(key_x)
        key_eta, subkey_eta = random.split(key_eta)

        batch_size = min(block_size, total_samples - i)
        if dist == "gaussian":
        	x = sample_gaussian(subkey_x, batch_size, p, variance=1.0)
        elif dist == "student":
        	x = sample_student(subkey_x, batch_size, p, df)
        eta = jnp.sqrt(0.1) * random.normal(subkey_eta, shape=(batch_size,))

        t = 2 * jnp.sum(x, axis=1) + eta
        epsilon = 0.5 * eta
        phi_t = jnp.sin(jnp.expand_dims(t, axis=1) + jnp.arange(1, d+1))
        y = phi_t @ omega + epsilon

        yield x[:ratio_N], x[ratio_N:], t[:ratio_N], y[ratio_N:]
