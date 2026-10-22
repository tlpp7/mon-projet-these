import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 50, 1000)
# Système dynamique couplé simulant l'orbite de phase de la solution exacte
x_trajectory = np.exp(-0.1 * t) * np.cos(t)
y_velocity = -0.1 * np.exp(-0.1 * t) * np.cos(t) - np.exp(-0.1 * t) * np.sin(t)

# Ajout d'une perturbation limite près du point d'équilibre (Bruit Quantique)
x_perturbed = x_trajectory + 0.05 * np.random.normal(size=len(t)) * np.exp(-0.05 * t)
y_perturbed = y_velocity + 0.05 * np.random.normal(size=len(t)) * np.exp(-0.05 * t)

plt.figure(figsize=(7, 6))
plt.plot(x_trajectory, y_velocity, 'k--', label="Orbite Théorique Stable", alpha=0.5)
plt.plot(x_perturbed, y_perturbed, 'teal', label="Orbite Réelle avec Bruit Quantique")
plt.scatter(0, 0, color='red', s=100, zorder=5, label="Attracteur Fixe Absolu $x_0^\dagger$")

plt.title("8. Portrait de Phase Asymptotique et Stabilité de l'Attracteur")
plt.xlabel("Résidu de la Contrainte Inversée")
plt.ylabel("Vitesse de Variation Variationnelle")
plt.grid(True, linestyle=':')
plt.legend()
plt.show()
