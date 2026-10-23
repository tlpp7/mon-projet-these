import numpy as np
import matplotlib.pyplot as plt

def compute_gcv_score(alpha_space, n_data=500):
    """
    Simule la fonction de score GCV pour un problème inverse régularisé.
    Le minimum représente la balance optimale bruit/biais.
    """
    np.random.seed(42)
    # Modélisation analytique de la courbe de score GCV (U-shape)
    residual_power = 0.05 / (alpha_space + 1e-5)**0.4
    degrees_of_freedom_influence = (1.0 - 0.1 * np.log10(alpha_space + 1e-5))**2
    
    score = (residual_power / degrees_of_freedom_influence) + 0.02 * alpha_space**1.2
    return score + np.random.normal(0, 0.001, len(alpha_space))

if __name__ == "__main__":
    alphas = np.logspace(-4, 1, 200)
    gcv_scores = compute_gcv_score(alphas)
    
    # Recherche du alpha optimal (minimum du score)
    opt_idx = np.argmin(gcv_scores)
    alpha_opt = alphas[opt_idx]
    
    plt.figure(figsize=(7, 4.5))
    plt.loglog(alphas, gcv_scores, color='purple', linewidth=2, label="Score GCV $(\alpha)$")
    plt.axvline(x=alpha_opt, color='black', linestyle=':', linewidth=1.5)
    plt.scatter(alpha_opt, gcv_scores[opt_idx], color='crimson', s=60, zorder=5, label=f"$\\alpha^*$ Optimal = {alpha_opt:.4f}")
    
    plt.title("20. Choix Automatique du Paramètre par Score GCV")
    plt.xlabel(r"Paramètre de Régularisation $\alpha$ (Échelle Log)")
    plt.ylabel("Score GCV Extrait")
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
