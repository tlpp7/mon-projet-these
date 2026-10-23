                                RÉSUMÉ DE THÈSE
                                
Titre : Régularisation variationnelle, transport optimal et convergence des opérateurs 
        prox-inverses sous contraintes non convexes de Nash.

Mots-clés : Problèmes inverses, Optimisation non convexe, Transport Optimal, Homologie 
            Persistante, Opérateurs Neuronaux, Calcul Quantique Variationnel.

Résumé :
Cette thèse s'attaque au verrou scientifique majeur des problèmes inverses non linéaires 
mal posés en dimension infinie, lorsque l'espace des solutions admissibles est restreint 
par des contraintes géométriques hautement non convexes. Traditionnellement traitées comme 
des obstacles computationnels provoquant des stagnations dans des minima locaux artificiels, 
nous démontrons dans ce travail que ces contraintes — modélisées sous la forme de variétés 
sous-analytiques de Nash — peuvent être transmutées en forces géodésiques motrices.

Nous introduisons un framework unifié articulé autour de trois piliers disrupteurs :
1) La formulation d'un flux de gradient proximal de Jordan-Kinderlehrer-Otto (JKO) dans 
   l'espace de Wasserstein $W_2$. En exploitant la régularisation entropique de Sinkhorn, 
   nous prouvons la convexification globale du paysage énergétique au sens d'Alexandrov.
2) L'injection d'invariants topologiques issus de l'Analyse Topologique des Données (TDA). 
   Le développement d'un algorithme de rétropropagation du gradient de persistance 
   (PH-Grad) garantit la préservation stricte des nombres de Betti ($\beta_1$) face à 
   des régimes de bruits stochastiques extrêmes, avant le franchissement du seuil de 
   catastrophe topologique de René Thom.
3) L'accélération numérique via des architectures hybrides. Nous déployons des schémas 
   multi-grilles micro/macro stables, des opérateurs neuronaux de Fourier contraints 
   (C-FNO) et un algorithme proximal quantique variationnel (Q-Prox) exploitant la règle 
   du saut de paramètre sur QPU. 

Les théorèmes de double $\Gamma$-convergence presque sûre établis sont validés par une 
bibliothèque de simulations numériques ouvertes, atteignant des précisions de reconstruction 
inégalées de l'ordre de $10^{-7}$ sous une complexité quasi-linéaire $\mathcal{O}(N \log N)$.

─────────────────────────────────────────────────────────────────────────────────────────

                               THESIS ABSTRACT
                                
Title: Variational regularization, optimal transport, and convergence of prox-inverse 
       operators under non-convex Nash constraints.

Keywords: Inverse problems, Non-convex optimization, Optimal Transport, Persistent 
          Homology, Neural Operators, Variational Quantum Computing.

Abstract:
This thesis addresses a major scientific bottleneck in infinite-dimensional non-linear 
ill-posed inverse problems where the space of admissible solutions is restricted by highly 
non-convex geometric constraints. Mathematically framed as subanalytic Nash manifolds, 
these constraints have historically been viewed as computational obstacles trapping algorithms 
in artificial local minima. This work demonstrates that they can be transmuted into driving 
geodesic forces.

We introduce a unified framework built upon three disruptive pillars:
1) The formulation of a Jordan-Kinderlehrer-Otto (JKO) proximal gradient flow within the 
   Wasserstein space $W_2$. Utilizing Sinkhorn's entropic regularization, we prove the 
   global convexification of the energy landscape in the Alexandrov sense.
2) The integration of topological invariants derived from Topological Data Analysis (TDA). 
   The development of a persistent gradient backpropagation algorithm (PH-Grad) guarantees 
   the strict preservation of Betti numbers ($\beta_1$) under extreme stochastic noise 
   regimes, well before reaching René Thom's critical topological catastrophe threshold.
3) Numerical acceleration via hybrid architectures. We deploy stable macro/micro multigrid 
   schemes, Constrained Fourier Neural Operators (C-FNO), and a variational quantum proximal 
   algorithm (Q-Prox) leveraging the parameter-shift rule on QPUs.

The proven almost-sure double $\Gamma$-convergence theorems are validated by an open-source 
numerical simulation suite, achieving unprecedented reconstruction accuracies down to $10^{-7}$ 
with a quasi-linear complexity of $\mathcal{O}(N \log N)$.
