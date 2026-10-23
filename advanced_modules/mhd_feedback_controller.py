"""
Contrôleur Prédictif d'Instabilité Magnétohydrodynamique (MHD) & Smart Grids.
Régulation géométrique active guidée par l'inversion d'entropie topologique.
"""

import numpy as np

class GeometricFeedbackController:
    def __init__(self, alert_threshold=0.60, critical_threshold=0.85):
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.control_history = []
        self.state_history = []

    def compute_control_action(self, current_state, topological_entropy):
        """
        Calcule la correction de contrôle u(t) via une fonctionnelle de Lyapunov contrôlée.
        """
        self.state_history.append(current_state)
        
        # 1. Régime nominal : contrôle passif minimal
        if topological_entropy < self.alert_threshold:
            u_t = - 0.05 * current_state
            mode = "NOMINAL"
            
        # 2. Régime d'alerte précoce : injection de stabilisation géodésique
        elif self.alert_threshold <= topological_entropy < self.critical_threshold:
            severity = (topological_entropy - self.alert_threshold) / (self.critical_threshold - self.alert_threshold)
            # Contre-terme non linéaire proportionnel à l'instabilité
            u_t = - 0.8 * current_state - 1.5 * severity * np.sign(current_state)
            mode = "ALERTE_ACTIVE"
            
        # 3. Régime de crise critique : Fronde de sécurité d'urgence
        else:
            u_t = - 4.0 * current_state
            mode = "ARRET_D_URGENCE_FRONDE"
            
        self.control_history.append(u_t)
        return u_t, mode

if __name__ == "__main__":
    controller = GeometricFeedbackController()
    
    # Simulation d'une trajectoire approchant la catastrophe
    states = [0.8, 0.85, 0.92, 0.98]
    entropies = [0.45, 0.65, 0.78, 0.88] # Croissance rapide
    
    print("✓ Simulation du Contrôleur Prédictif Géométrique :")
    for t, (s, e) in enumerate(zip(states, entropies)):
        action, regime = controller.compute_control_action(s, e)
        print(f"  t={t}s | État={s:.2f} | Entropie={e:.2f} | Mode: {regime:<22} | Action u(t)={action:.3f}")
