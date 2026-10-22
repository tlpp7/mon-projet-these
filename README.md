# 🏛️ Advanced Constrained Inverse Problems Framework

[![Python Version](https://shields.io)](https://python.org)
[![License: MIT](https://shields.io)](https://opensource.org)

Ce dépôt héberge le moteur algorithmique et les simulations de validation de ma thèse de doctorat intitulée : **"Régularisation variationnelle, transport optimal et convergence des opérateurs prox-inverses sous contraintes non convexes de Nash"**.

L'objectif fondamental de ce projet est de surmonter le fléau de la contrainte géométrique inverse non convexe en la convertissant en une force géodésique motrice par le biais du transport optimal et de la topologie algébrique.

---

## 🔬 Architecture Synthétique du Framework

Le système résout l'équation non linéaire mal posée \(y^\delta = F(x) + \eta\) en couplant l'action de **16 agents mathématiciens virtuels** organisés selon le pipeline distribué suivant :

```text
 [ Données Bruitées yᵟ ] ──> Échantillonnage Dynamique Variance Réduite (A-IS)
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        ▼                                                           ▼
💻 Pipeline HPC / GPU                                     ⚛️ Accélérateur QPU
• Cartes de Diffusion (Linéarisation)                     • Ansatz Variationnel (Q-Prox)
• Flux JKO Entropique (Wasserstein W₂)                    • Évaluation des inclusions fines
        │                                                           │
        └─────────────────────────────┬─────────────────────────────┘
                                      ▼
                       [ Couplage Central Variationnel ]
                                      │
                                      ▼
                       📈 Pipeline Topologique (TDA)
                       • Complexes Simpliciaux & Filtration H₀/H₁
                       • Rétropropagation de Potentiel (PH-Grad)
                                      │
                                      ▼
                       [ ATTRACTEUR GLOBAL DE NASH x₀† ]
```

---

## 📊 Aperçu des Résultats Mathématiques

En exécutant notre script de simulation complet, le framework génère une planche d'analyse avancée mettant en évidence les comportements asymptotiques et les transitions de phase :

*   **Effet Fronde Topologique :** Inversion de l'action de la contrainte (bascule super-exponentielle vers l'optimum global).
*   **Seuil de Catastrophe :** Analyse limite de l'effondrement entropique des structures géométriques sous l'action d'un bruit critique.
*   **Support Compact JKO :** Concentration de la mesure de transport de masse sur une variété parabolique stricte.
*   **Diffusion Maps :** Dépliement non linéaire de l'espace de configuration pour s'affranchir des minima locaux.

---

## 💻 Installation & Utilisation Rapide

### 1. Configurer l'environnement
```bash
pip install -r requirements.txt
```

### 2. Exécuter la simulation et générer le tableau de bord
```bash
python simulation_complete.py
```

---

## 🛠️ Spécifications des Algorithmes Principaux

1.  **S-JKO Flow (`core/agents_math.py`) :** Implémentation du flux de Jordan-Kinderlehrer-Otto via l'algorithme de Sinkhorn à régularisation entropique pour l'advection rapide des mesures.
2.  **Diffusion Embedding (`core/agents_math.py`) :** Décomposition spectrale du Laplacien de graphe normalisé pour transformer des problèmes non convexes rigides en géodésiques Riemanniennes lisses.
