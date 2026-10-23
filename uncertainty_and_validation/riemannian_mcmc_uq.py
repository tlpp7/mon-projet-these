"""
Riemannian Manifold Hamiltonian Monte Carlo (RMHMC) pour la quantification d'incertitude bayésienne
sur variété riemannienne d'information.
"""

import numpy as np

class RiemannianWassersteinSampler:
    def __init__(self, target_log_prob_fn, metric_tensor_fn, dim=2):
        self.log_prob_fn = target_log_prob_fn
        self.metric_tensor_fn = metric_tensor_fn
        self.dim = dim

    def sample_chain(self, x_init, n_samples=300, step_size=0.08, n_leapfrog=6):
        samples = [x_init]
        x_curr = x_init.copy()
        accepted = 0

        for s in range(n_samples):
            # 1. Tenseur métrique G(x) et son inverse
            G = self.metric_tensor_fn(x_curr)
            G_inv = np.linalg.inv(G)
            
            # 2. Échantillonnage de la quantité de mouvement p ~ N(0, G(x))
            p_curr = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=G)
            
            # Énergie Hamiltonienne initiale : H = -log_prob + 0.5 * p^T * G^-1 * p + 0.5 * log(det(G))
            H_init = -self.log_prob_fn(x_curr) + 0.5 * (p_curr @ G_inv @ p_curr) + 0.5 * np.linalg.slogdet(G)[1]
            
            x_prop = x_curr.copy()
            p_prop = p_curr.copy()
            
            # 3. Intégrateur symplectique géodésique (Leapfrog sur variété)
            for _ in range(n_leapfrog):
                # Demi-pas d'impulsion
                grad_V = (self.log_prob_fn(x_prop + 1e-5) - self.log_prob_fn(x_prop - 1e-5)) / (2 * 1e-5)
                p_prop += 0.5 * step_size * grad_V
                # Pas de position le long de la géodésique
                G_prop = self.metric_tensor_fn(x_prop)
                x_prop += step_size * (np.linalg.inv(G_prop) @ p_prop)
                # Demi-pas d'impulsion
                p_prop += 0.5 * step_size * grad_V

            # Énergie finale
            G_final = self.metric_tensor_fn(x_prop)
            G_final_inv = np.linalg.inv(G_final)
            H_final = -self.log_prob_fn(x_prop) + 0.5 * (p_prop @ G_final_inv @ p_prop) + 0.5 * np.linalg.slogdet(G_final)[1]
            
            # 4. Critère d'acceptation de Metropolis-Hastings
            alpha = np.exp(min(0.0, -(H_final - H_init)))
            if np.random.rand() < alpha:
                x_curr = x_prop
                accepted += 1
                
            samples.append(x_curr.copy())

        acceptance_rate = accepted / n_samples
        return np.array(samples), acceptance_rate

if __name__ == "__main__":
    def toy_log_prob(x):
        # A posteriori non convexe avec anneau de vraisemblance
        r = np.linalg.norm(x)
        return -0.5 * ((r - 1.5)**2 / 0.1)

    def toy_metric(x):
        # Métrique de Fisher-Rao locale
        base = 1.0 + 0.5 * (x[0]**2 + x[1]**2)
        return np.array([[base, 0.1], [0.1, base]])

    sampler = RiemannianWassersteinSampler(toy_log_prob, toy_metric, dim=2)
    chain, acc = sampler.sample_chain(np.array([1.2, 0.2]), n_samples=150)
    print(f"✓ Échantillonnage RMHMC terminé. Taux d'acceptation sur variété : {acc*100:.1f}%")
