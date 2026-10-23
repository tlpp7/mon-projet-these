# ==============================================================================
# Flot de Gradient Wasserstein JKO continu en Julia (High-Performance SciML)
# ==============================================================================

using DifferentialEquations
using LinearAlgebra
using ForwardDiff

"""
Potentiel d'énergie libre non-convexe avec barrière métastable
"""
function free_energy_potential(x)
    return 0.5 * dot(x, x) - 1.5 * exp(-norm(x .- [1.0, 1.0])^2 / 0.4) - 2.0 * exp(-norm(x .+ [1.2, 1.2])^2 / 0.5)
end

"""
Dynamique de dérive JKO guidée par la métrique de Wasserstein W2
"""
function jko_drift!(dx, x, p, t)
    # Dérivation automatique exacte du gradient d'énergie
    grad_V = ForwardDiff.gradient(free_energy_potential, x)
    
    # Invariant de fronde topologique (force motrice orthogonale)
    slingshot_correction = [-x[2], x[1]] * 0.4 * exp(-t)
    
    # Équation d'advection pure
    dx .= -grad_V .+ slingshot_correction
end

"""
Diffusion brownienne régularisée
"""
function jko_diffusion!(dx, x, p, t)
    sigma = 0.15 # Bruit thermique sous-critique
    dx[1] = sigma
    dx[2] = sigma
end

# Conditions initiales et horizon temporel
x0 = [1.8, 1.8] # Démarrage dans le bassin métastable
tspan = (0.0, 5.0)

# Définition du problème SDE (Itô)
prob = SDEProblem(jko_drift!, jko_diffusion!, x0, tspan)
sol = solve(prob, SRIW1(), dt=0.01, adaptive=true)

println("✓ Intégration SciML terminée avec succès. Points temporels calculés : ", length(sol.t))
