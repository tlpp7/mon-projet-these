import numpy as np
import matplotlib.pyplot as plt

def simuler_propagation_front(n_space=200, n_time=5):
    """
    Modélise la formation et la dérive d'une singularité géométrique non linéaire
    (type onde de choc d'une contrainte inversée évolutive).
    """
    x = np.linspace(-3, 3, n_space)
    trajectoires = []
    
    for t in range(n_time):
        # Évolution d'une fonction hyperbolique simulant le pincement du front
        pente = 1.0 + t * 1.5
        front = -np.tanh(pente * (x - 0.2 * t))
        trajectoires.append(front)
        
    return x, trajectoires

if __name__ == "__main__":
    space_axis, temps_series = simuler_propagation_front()
    
    plt.figure(figsize=(8, 5))
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(temps_series)))
    
    for idx, front_t in enumerate(temps_series):
        plt.plot(space_axis, front_t, color=colors[idx], linewidth=1.5, 
                 label=f"Étape Temporelle $t_{idx}$ (Déformation de la Variété)")
        
    plt.axvline(x=0.8, color='black', linestyle='--', alpha=0.3, label="Point d'Attraction Final")
    plt.title("23. Suivi de Singularité Non Linéaire en Régime Transitoire")
    plt.xlabel("Coordonnée Géométrique Intrinsèque")
    plt.ylabel("Profil Spatial du Front Inversé")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()
