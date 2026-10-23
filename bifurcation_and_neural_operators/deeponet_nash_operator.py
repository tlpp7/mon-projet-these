"""
Deep Operator Network (DeepONet) pour l'apprentissage des équilibres de Nash couplés.
Architecture Branch Net (profil de contrainte) + Trunk Net (espace des coordonnées).
"""

import torch
import torch.nn as nn

class BranchNet(nn.Module):
    """Réseau de branche : encode la stratégie de l'agent adverse u(x) discrétisée en m capteurs"""
    def __init__(self, m_sensors=50, p_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m_sensors, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, p_dim)
        )
    def forward(self, u):
        return self.net(u)

class TrunkNet(nn.Module):
    """Réseau de tronc : encode les coordonnées physiques d'évaluation y"""
    def __init__(self, in_dim=1, p_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, p_dim)
        )
    def forward(self, y):
        return self.net(y)

class DeepONetNash(nn.Module):
    def __init__(self, m_sensors=50, p_dim=64):
        super().__init__()
        self.branch = BranchNet(m_sensors, p_dim)
        self.trunk = TrunkNet(1, p_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors, y_coords):
        # u_sensors: [batch, m_sensors]
        # y_coords: [batch, num_points, 1]
        B = self.branch(u_sensors) # [batch, p_dim]
        T = self.trunk(y_coords)   # [batch, num_points, p_dim]
        
        # Produit scalaire de dualité de Hilbert + biais scalaire
        out = torch.einsum("bp,bnp->bn", B, T) + self.bias
        return out

if __name__ == "__main__":
    model = DeepONetNash(m_sensors=30, p_dim=32)
    fake_strategy_adversaire = torch.randn(8, 30)
    fake_grid_evaluation = torch.linspace(-1, 1, 100).view(1, 100, 1).repeat(8, 1, 1)
    
    nash_response = model(fake_strategy_adversaire, fake_grid_evaluation)
    print("✓ DeepONet Nash : Tenseur de réaction généré :", nash_response.shape)
