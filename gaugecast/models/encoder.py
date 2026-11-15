import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x_hist_enc):
        _, (h_n, _) = self.lstm(x_hist_enc)
        context = h_n[-1]
        return context
