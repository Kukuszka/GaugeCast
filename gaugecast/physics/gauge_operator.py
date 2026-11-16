import torch
import torch.nn as nn


class GaugeNet(nn.Module):
    def __init__(self, context_dim=64, dec_dim=64, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + dec_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tau, x, context, dec_feat):
        B, P, _ = tau.shape
        c = context.unsqueeze(1).expand(B, P, context.shape[-1])
        inp = torch.cat([tau, x, c, dec_feat], dim=-1)
        phi = x + 0.1 * torch.tanh(self.net(inp))
        return phi


def compute_gauge_residual(phi_tau, phi_x, phi_xx, a, nu):
    B = phi_tau + a * phi_x - nu * phi_xx
    return B
