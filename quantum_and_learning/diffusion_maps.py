import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def compute_diffusion_map(X_data, sigma=0.2, n_components=2):
    """ Calcule le plongement de diffusion non linéaire pour simplifier la contrainte """
    dists = np.sum((X_data[:, np.newaxis, :] - X_data[np.newaxis, :, :])**2, axis=-1)
    W = np.exp(-dists / (2 * sigma**2))
    D = np.diag(1.0 / np.sum(W, axis=1))
    P_matrix = D @ W
    
    # Décomposition spectrale
    eigvals, eigvecs = la.eigh(P_matrix)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx[1:n_components+1]] * eigvals[idx[1:n_components+1]]

if __name__ == "__main__":
    np.random.seed(42)
    theta = np.linspace(-np.pi, np.pi, 300)
    z_coord = np.sin(2 * theta) + np.random.normal(0, 0.05, 300)
    X_manifold = np.stack([np.cos(theta), np.sin(theta), z_coord], axis=-1)

    embedding = compute_diffusion_map(X_manifold, sigma=0.3)

    plt.figure(figsize=(6, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=theta, cmap='viridis', s=25, edgecolor='k', alpha=0.8)
    plt.title("13. Linéarisation Riemannienne par Plongement de Diffusion")
    plt.xlabel("Coordonnée de Diffusion $\psi_1$")
    plt.ylabel("Coordonnée de Diffusion $\psi_2$")
    plt.colorbar(label="Paramétrage intrinsèque de la contrainte")
    plt.grid(True, ls=":")
    plt.show()
