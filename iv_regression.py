import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cholesky
from jax import grad
from kernel import Kernel
from utils import gd_update, backtracking_line_search, feature_matrix

class IVRegression:
    def __init__(self, x, t, x_tilde, y_tilde, lambda_reg, d, kernel: Kernel, baseline):
        self.x = x
        self.t = t
        self.x_tilde = x_tilde
        self.y_tilde = y_tilde
        self.lambda_reg = lambda_reg
        self.d = d
        self.kernel = kernel
        self.baseline = baseline
    
    def Kbar_C_hat(self):
        F = feature_matrix(self.t, self.d)
        if self.baseline == True:
            Xi, Xi_tilde = self.kernel.Xi_Xitilde(self.x, self.x_tilde)
            n, D = Xi.shape
            reg_term = n * self.lambda_reg * jnp.eye(D)
            K_reg = self.kernel.kernel(self.x, self.x_tilde) + reg_term
            L = cholesky(K_reg, lower=True)
            Xi_t_F = Xi.T @ F
            K_reg_inv_Xi_t_F = cho_solve((L, True), Xi_t_F)
            return Xi_tilde @ K_reg_inv_Xi_t_F
        else:
            n = (self.x).shape[0]
            Kbar = self.kernel.kernel(self.x_tilde, self.x)
            K = self.kernel.kernel(self.x, self.x)
            reg_term = n * self.lambda_reg * jnp.eye(n)
            K_reg = K + reg_term
            L = cholesky(K_reg, lower=True)
            return Kbar @ cho_solve((L, True), F)
    
    def value(self, omega):
        m = self.y_tilde.shape[0]
        term = self.Kbar_C_hat() @ omega - self.y_tilde
        value = (1 / (2 * m)) * term @ term
        return value
    
    def grad(self, omega):
        gradient = grad(self.value)
        return gradient(omega)

    def optimize(self, omega_initial, epsilon):
        omega_trajectory = []
        value_trajectory = []
        gradient_trajectory = []
        omega = omega_initial
        value = self.value(omega)
        gradient = self.grad(omega)
        omega_trajectory.append(omega)
        value_trajectory.append(value)
        gradient_trajectory.append(gradient)
        while jnp.linalg.norm(gradient) > epsilon:
            eta = backtracking_line_search(self.value, self.grad, omega, eta=1.0, tau=0.5, c=1e-4)
            omega = gd_update(omega, eta, gradient)
            omega_trajectory.append(omega)
            value = self.value(omega)
            gradient = self.grad(omega)
            value_trajectory.append(value)
            gradient_trajectory.append(gradient)
        return omega_trajectory, value_trajectory, gradient_trajectory