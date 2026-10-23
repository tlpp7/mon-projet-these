import numpy as np
import matplotlib.pyplot as plt

def simuler_splitting_alterne(n_steps):
    """
    Simule la décroissance du résidu variationnel sous splitting proximal alterné.
    """
    np.random.seed(24)
    steps = np.arange(n_steps)
    # Décroissance globale modulée par les sauts alternés entre facettes de Nash
    residu = 0.7 * np.exp(-0.02 * steps) + 0.05 * np.abs(np.sin(0.15 * steps)) * np.exp(-0.005 * steps)
    return residu

if __name__ == "__main__":
    it_steps = 400
    residu_splitting = simuler_splitting_alterne(it_steps)
    
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(it_steps), residu_splitting, color='darkorange', linewidth=2, label="Splitting Proximal Alterné")
    plt.axhline(y=1e-5, color='gray', linestyle='--', alpha=0.7, label="Seuil de Tolérance $\tau$")
    plt.title("19. Convergence du Schéma de Splitting sous Contraintes Rigides")
    plt.xlabel("Nombre de Sous-Itérations de Splitting")
    plt.ylabel("Norme du Résidu Operatoriel $\|F(x_k) - y^\delta\|$")
    plt.yscale('log')
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
