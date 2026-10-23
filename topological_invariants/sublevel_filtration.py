import numpy as np
import matplotlib.pyplot as plt

def generate_persistence_data():
    """ Simule l'extraction des points de naissance et de mort (H0) """
    np.random.seed(101)
    t = np.linspace(0, 10, 500)
    clean_signal = np.sin(t) + 0.5 * np.sin(2.5 * t)
    noisy_signal = clean_signal + np.random.normal(0, 0.3, len(t))

    births, deaths = [], []
    for i in range(1, len(noisy_signal) - 1):
        if noisy_signal[i] < noisy_signal[i-1] and noisy_signal[i] < noisy_signal[i+1]:
            births.append(noisy_signal[i])
            peak = np.max(noisy_signal[i:])
            deaths.append(peak if peak > noisy_signal[i] else noisy_signal[i] + 0.5)
            
    return np.array(births), np.array(deaths)

if __name__ == "__main__":
    births, deaths = generate_persistence_data()

    plt.figure(figsize=(6, 6))
    plt.scatter(births, deaths, color='darkblue', s=40, zorder=3, label='Invariants Topologiques')
    
    # Ligne diagonale de mort immédiate
    min_val = min(np.min(births), np.min(deaths))
    max_val = max(np.max(births), np.max(deaths))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Seuil du Bruit (Mort immédiate)')
    plt.fill_between([min_val, max_val], [min_val, max_val], max_val, color='gray', alpha=0.1)

    plt.title("17. Diagramme de Persistance Topologique (Filtration H0)")
    plt.xlabel("Seuil de Naissance (Birth)")
    plt.ylabel("Seuil de Mort (Death)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()
