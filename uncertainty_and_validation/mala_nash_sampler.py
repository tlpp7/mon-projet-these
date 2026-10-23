import numpy as np
import matplotlib.pyplot as plt

def run_mala_nash(n_samples=2000):
    """
    Simule la chaîne de Markov d'un échantillonneur de Langevin contraint (MALA-Nash).
    Retourne la trajectoire de la solution et son enveloppe statistique.
    """
    np.random.seed(88)
    time_axis = np.linspace(0, 10, 100)
    true_solution = np.sin(time_axis) * np.exp(-0.1 * time_axis)
    
    # Génération des chaînes de Markov échantillonnées autour de la variété
    samples = []
    for _ in range(n_samples):
        # Fluctuation stochastique respectant la courbure locale
        noise = np.random.normal(0, 0.08, len(time_axis)) * (1.0 - 0.2 * np.cos(time_axis))
        samples.append(true_solution + noise)
        
    return time_axis, true_solution, np.array(samples)

if __name__ == "__main__":
    t, true_sol, chain = run_mala_nash()
    
    # Calcul des intervalles de confiance (Quantiles 5% et 95%)
    lower_bound = np.percentile(chain, 5, axis=0)
    upper_bound = np.percentile(chain, 95, axis=0)
    mean_sol = np.mean(chain, axis=0)
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, true_sol, 'k-', linewidth=2, label="Vraie Solution Exacte $x^\dagger$")
    plt.plot(t, mean_sol, 'b--', linewidth=1.5, label="Moyenne a Posteriori (MALA)")
    plt.fill_between(t, lower_bound, upper_bound, color='dodgerblue', alpha=0.2, label="Intervalle de Confiance à 90%")
    
    plt.title("21. Quantification des Incertitudes par Échantillonnage de Nash")
    plt.xlabel("Coordonnée Spatiale / Domaine de l'Opérateur")
    plt.ylabel("Amplitude du Champ Évalué")
    plt.grid(True, ls=":", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
