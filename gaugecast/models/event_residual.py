import torch
import torch.nn as nn


class EventResidualHead(nn.Module):
    """
    C1 residual branch with leaky event accumulation.
    """
    def __init__(self, context_dim=64, dec_dim=64, hidden_dim=64):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(context_dim + dec_dim + 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim + dec_dim + 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tau, x, fut_seq, context, dec_feat):
        B, P, _ = tau.shape
        c = context.unsqueeze(1).expand(B, P, context.shape[-1])
        inp = torch.cat([tau, x, fut_seq, c, dec_feat], dim=-1)

        delta_raw = self.delta_net(inp)
        gate = torch.sigmoid(self.gate_net(inp))
        delta_event = gate * delta_raw

        event_list = []
        running = torch.zeros_like(delta_event[:, 0:1, :])
        for k in range(delta_event.size(1)):
            running = 0.60 * running + delta_event[:, k:k+1, :]
            event_list.append(running)
        event_residual = torch.cat(event_list, dim=1)

        return delta_event, event_residual, gate
