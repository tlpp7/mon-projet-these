"""
Flot de Gradient Wasserstein-Fisher-Rao (WFR / Hellinger-Kantorovich).
Permet le transport optimal avec création/perte de masse pour la physique des plasmas et la biochimie.
"""

import numpy as np
import scipy.ndimage as ndimage

class UnbalancedWFRFlow:
    def __init__(self, grid_size=200, domain=(-3.0, 3.0), alpha=1.0, beta=1.0):
        """
        alpha : Poids du transport spatial (Wasserstein W2)
        beta  : Poids de la réaction de source/perte de masse (Fisher-Rao)
        """
        self.grid_size = grid_size
        self.x = np.linspace(domain[0], domain[1], grid_size)
        self.dx = self.x[1] - self.x[0]
        self.alpha = alpha
        self.beta = beta

    def energy_functional(self, rho):
        """Énergie libre V(x) + régularisation entropique non convexe"""
        V = 0.5 * (self.x**2 - 1.0)**2  # Double puits de potentiel
        entropy = np.where(rho > 1e-12, rho * np.log(rho + 1e-12), 0.0)
        return np.sum((V * rho + 0.1 * entropy) * self.dx)

    def first_variation(self, rho):
        """Dérivée variationnelle dF/drho"""
        V = 0.5 * (self.x**2 - 1.0)**2
        log_term = np.log(np.maximum(rho, 1e-12)) + 1.0
        return V + 0.1 * log_term

    def step(self, rho, dt=0.005):
        """Un pas de temps par opérateur-splitting WFR"""
        mu = self.first_variation(rho)
        
        # 1. Étape de Transport Spatial (Wasserstein W2)
        # Flux d'advection : J = - rho * grad(mu)
        grad_mu = np.gradient(mu, self.dx)
        flux = - rho * grad_mu
        div_flux = - np.gradient(flux, self.dx)
        
        # 2. Étape de Réaction / Changement de masse (Fisher-Rao)
        # Terme de croissance/annihilation
        reaction = - (mu - np.sum(mu * rho) * self.dx) * rho
        
        # Mise à jour combinée
        rho_next = rho + dt * (self.alpha * div_flux + self.beta * reaction)
        rho_next = np.maximum(rho_next, 0.0)
        return rho_next

if __name__ == "__main__":
    solver = UnbalancedWFRFlow(grid_size=150)
    # Condition initiale unimodale asymétrique
    rho_0 = np.exp(- (solver.x + 1.5)**2 / 0.2)
    rho_0 = rho_0 / (np.sum(rho_0) * solver.dx)
    
    rho = rho_0.copy()
    for _ in range(50):
        rho = solver.step(rho)
        
    mass_initial = np.sum(rho_0) * solver.dx
    mass_final = np.sum(rho) * solver.dx
    print(f"✓ Transport WFR simulé. Masse initiale = {mass_initial:.4f}, Masse finale dynamique = {mass_final:.4f}")
