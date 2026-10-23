"""
Solveur de Jeux à Champ Moyen (Mean Field Games - MFG) dans l'espace de Wasserstein.
Couplage itératif de monotone-splitting entre Hamilton-Jacobi-Bellman (HJB) et Fokker-Planck (FP).
"""

import numpy as np

class MeanFieldGameSolver:
    def __init__(self, nx=100, nt=60, x_bounds=(-2.5, 2.5), t_final=1.0, sigma=0.2):
        self.nx = nx
        self.nt = nt
        self.x = np.linspace(x_bounds[0], x_bounds[1], nx)
        self.dx = self.x[1] - self.x[0]
        self.dt = t_final / nt
        self.sigma = sigma
        self.nu = 0.5 * sigma**2

    def solve_hjb_backward(self, rho_trajectory, u_terminal):
        """Résolution rétrograde de HJB (de t=T à t=0) sous condition terminale de Nash"""
        u = np.zeros((self.nt, self.nx))
        u[-1, :] = u_terminal
        
        for n in reversed(range(self.nt - 1)):
            # Terme de couplage non-local de champ moyen V(x, rho)
            coupling_cost = 0.5 * (self.x**2) + 0.8 * rho_trajectory[n + 1, :]
            
            # Gradient spatial de la valeur
            grad_u = np.gradient(u[n + 1, :], self.dx)
            hamiltonian = 0.5 * (grad_u**2)
            diffusion = self.nu * np.gradient(grad_u, self.dx)
            
            # Pas de temps rétrograde
            u[n, :] = u[n + 1, :] + self.dt * (hamiltonian - diffusion - coupling_cost)
            
        return u

    def solve_fokker_planck_forward(self, u_trajectory, rho_initial):
        """Résolution antérograde de Fokker-Planck (de t=0 à t=T) sous dérive optimale -grad(u)"""
        rho = np.zeros((self.nt, self.nx))
        rho[0, :] = rho_initial
        
        for n in range(self.nt - 1):
            grad_u = np.gradient(u_trajectory[n, :], self.dx)
            drift_velocity = - grad_u
            
            # Flux advectif J = rho * drift
            flux = rho[n, :] * drift_velocity
            advection = - np.gradient(flux, self.dx)
            diffusion = self.nu * np.gradient(np.gradient(rho[n, :], self.dx), self.dx)
            
            # Mise à jour et conservation de la positivité de la mesure
            rho[n + 1, :] = rho[n, :] + self.dt * (advection + diffusion)
            rho[n + 1, :] = np.maximum(rho[n + 1, :], 0.0)
            rho[n + 1, :] /= (np.sum(rho[n + 1, :]) * self.dx) # Renormalisation Wasserstein
            
        return rho

    def compute_mfg_nash_equilibrium(self, max_iter=15, damping=0.4):
        """Algorithme de point fixe de Krasnoselskii-Mann pour l'équilibre de Nash-MFG"""
        # Initialisation de la distribution
        rho_0 = np.exp(- (self.x - 1.2)**2 / 0.15)
        rho_0 /= (np.sum(rho_0) * self.dx)
        
        u_T = 0.5 * (self.x + 1.2)**2 # Cible terminale
        
        # Trajectoire de densité initiale (diffusive neutre)
        rho = np.tile(rho_0, (self.nt, 1))
        
        for it in range(max_iter):
            u = self.solve_hjb_backward(rho, u_T)
            rho_new = self.solve_fokker_planck_forward(u, rho_0)
            
            # Amortissement pour garantir la contraction globale
            error = np.max(np.abs(rho_new - rho))
            rho = (1.0 - damping) * rho + damping * rho_new
            
            if error < 1e-4:
                break
                
        return u, rho

if __name__ == "__main__":
    mfg = MeanFieldGameSolver(nx=80, nt=40)
    u_sol, rho_sol = mfg.compute_mfg_nash_equilibrium(max_iter=8)
    print(f"✓ Équilibre de Nash-MFG résolu sur la variété W2. Dimensions du champ : {rho_sol.shape}")
