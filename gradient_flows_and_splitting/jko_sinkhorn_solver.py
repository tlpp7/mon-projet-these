import numpy as np
import matplotlib.pyplot as plt

def solve_jko_sinkhorn(mu_source, nu_target, cost_matrix, epsilon=0.01, max_iter=150):
    """
    Calcule l'approximation JKO du transport de masse par régularisation entropique.
    """
    K = np.exp(-cost_matrix / epsilon)
    u = np.ones_like(mu_source) / len(mu_source)
    v = np.ones_like(nu_target) / len(nu_target)
    
    for _ in range(max_iter):
        u = mu_source / (K @ v + 1e-15)
        v = nu_target / (K.T @ u + 1e-15)
        
    plan_transport = np.diag(u) @ K @ np.diag(v)
    return plan_transport

if __name__ == "__main__":
    n_points = 100
    space = np.linspace(-2, 2, n_points)
    X, Y = np.meshgrid(space, space)
    C = (X - Y)**2  # Coût quadratique de transport
    
    # Simulation de deux profils de densité (Source vs Cible)
    mu = np.exp(-(space - 1.0)**2 / 0.1) + np.exp(-(space + 1.0)**2 / 0.1)
    mu /= np.sum(mu)
    nu = np.exp(-space**2 / 0.3)
    nu /= np.sum(nu)
    
    P = solve_jko_sinkhorn(mu, nu, C, epsilon=0.008)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(P, cmap='viridis', extent=[-2, 2, 2, -2])
    plt.title("18. Couplage Entropique du Flux Proximal-JKO")
    plt.xlabel("Espace Cible (Configurations Admissibles)")
    plt.ylabel("Espace Source (Solutions Intermédiaires)")
    plt.colorbar(label="Densité de Transport de Masse")
    plt.tight_layout()
    plt.show()
