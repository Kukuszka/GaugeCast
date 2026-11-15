import torch
import torch.nn as nn


class FutureRainDecoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.init_h = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, fut_seq, context):
        h0_single = torch.tanh(self.init_h(context)).unsqueeze(0)
        h0 = h0_single.repeat(self.num_layers, 1, 1)
        out, _ = self.gru(fut_seq, h0)
        return out


class LatentSolutionNet(nn.Module):
    def __init__(self, context_dim=64, dec_dim=64, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + dec_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, y_coord, tau, context, dec_feat):
        B, P, _ = y_coord.shape
        c = context.unsqueeze(1).expand(B, P, context.shape[-1])
        inp = torch.cat([y_coord, tau, c, dec_feat], dim=-1)
        v = self.net(inp)
        return v
