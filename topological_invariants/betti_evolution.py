import numpy as np
import matplotlib.pyplot as plt

def simuler_nombres_betti(bruit):
    """ Modélise la perte de structures topologiques H1 """
    # Profil de transition de phase topologique (fermeture brutale des trous)
    betti_1 = 12 / (1 + np.exp(6 * (bruit - 1.22)))
    return np.round(betti_1)

if __name__ == "__main__":
    bruit_axes = np.linspace(0, 2.0, 300)
    betti_path = [simuler_nombres_betti(b) for b in bruit_axes]

    plt.figure(figsize=(7, 4.5))
    plt.plot(bruit_axes, betti_path, 'crimson', linewidth=2.5, label="Nombre de Trous Détectés ($X_1$)")
    plt.axvspan(1.2, 2.0, color='red', alpha=0.1, label="Zone d'Effondrement Homologique")
    plt.title("16. Évolution des Invariants de Betti face au Bruit Critique")
    plt.xlabel("Intensité du Bruit Extérieur")
    plt.ylabel("Dimension Homologique $\beta_1$")
    plt.grid(True, ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
