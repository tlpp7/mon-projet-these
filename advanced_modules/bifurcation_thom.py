import numpy as np
import matplotlib.pyplot as plt

# Création du paysage énergétique de René Thom (Bifurcation en Cusp)
x = np.linspace(-2, 2, 200)
control_param = np.linspace(-1, 1, 5) # Évolution du couplage de la contrainte

plt.figure(figsize=(8, 5))
for u in control_param:
    # Équation de la catastrophe du Cusp : V(x) = x^4/4 + ux^2/2
    potential = (x**4) / 4 + (u * (x**2)) / 2
    plt.plot(x, potential, label=f"Paramètre de contrôle u = {u:.1f}")

plt.title("7. Paysage Énergétique sous Bifurcation de Catastrophe (Cusp)")
plt.xlabel("État du Système Inverse (x)")
plt.ylabel("Potentiel Topologique V(x)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
