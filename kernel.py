import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
from utils import sample_gaussian, gaussian_kernel_gram_matrix

class Kernel:
    def __init__(self, D, sigma, key, baseline):
        self.D = D
        self.sigma = sigma
        self.key = key
        self.baseline = baseline

    def Fourier_param(self, p):
        w = sample_gaussian(self.key, self.D, p, 1 / (self.sigma ** 2))
        b = random.uniform(self.key, shape=(self.D,), minval=0.0, maxval=2 * jnp.pi)
        return w, b

    def Xi_Xitilde(self, x, x_tilde):
        p = x.shape[1]
        w, b = self.Fourier_param(p)
        Xi = jnp.sqrt(2.0 / self.D) * jnp.cos(jnp.dot(x, w.T) + b)
        Xi_tilde = jnp.sqrt(2.0 / self.D) * jnp.cos(jnp.dot(x_tilde, w.T) + b)
        return Xi, Xi_tilde
    
    def kernel(self, x, x_tilde):
        if self.baseline == True:
            Xi = self.Xi_Xitilde(x, x_tilde)[0]
            return Xi.T @ Xi
        else:
            return gaussian_kernel_gram_matrix(x, x_tilde, self.sigma)