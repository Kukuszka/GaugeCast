import torch
import torch.nn as nn
import torch.nn.functional as F


class StorageDynamicsModule(nn.Module):
    """
    Storage-conditioned causal delay kernel module.
    """
    def __init__(self, context_dim=64, dec_dim=64, hidden_dim=64):
        super().__init__()

        self.init_net = nn.Sequential(
            nn.Linear(context_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.infil_net = nn.Sequential(
            nn.Linear(context_dim + dec_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.drain_net = nn.Sequential(
            nn.Linear(context_dim + dec_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.wet_gate_net = nn.Sequential(
            nn.Linear(context_dim + dec_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.delay_net = nn.Sequential(
            nn.Linear(context_dim + dec_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.base_gain_net = nn.Sequential(
            nn.Linear(context_dim + dec_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.vwc_obs_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def build_causal_kernel(self, delay_scale_seq):
        """
        delay_scale_seq: [B, P, 1]
        returns kernel K: [B, P, P], where row k only attends to j <= k
        """
        B, P, _ = delay_scale_seq.shape
        dtype = delay_scale_seq.dtype
        dev = delay_scale_seq.device

        idx = torch.arange(P, device=dev, dtype=dtype)
        lag = idx.view(P, 1) - idx.view(1, P)        # [P, P], k-j
        causal_mask = (lag >= 0)

        lag = lag.view(1, P, P).expand(B, P, P)
        delay = delay_scale_seq.expand(B, P, P).clamp_min(1e-6)

        logits = -lag / delay
        logits = logits.masked_fill(~causal_mask.view(1, P, P), -1e9)

        kernel = torch.softmax(logits, dim=-1)
        return kernel

    def forward(self, last_vwc, last_rain, fut_seq, context, dec_feat):
        B, P, _ = fut_seq.shape

        S_prev = F.softplus(self.init_net(torch.cat([context, last_vwc, last_rain], dim=-1)))

        S_list = []
        qbase_list = []
        infil_list = []
        drain_list = []
        wet_gate_list = []
        delay_scale_list = []
        quick_source_list = []

        for k in range(P):
            rain_k = fut_seq[:, k, :]
            dec_k = dec_feat[:, k, :]

            inp_cdS = torch.cat([context, dec_k, rain_k, S_prev], dim=-1)

            infil_coeff = torch.sigmoid(self.infil_net(inp_cdS))
            drain_coeff = F.softplus(self.drain_net(inp_cdS))
            wet_gate = torch.sigmoid(self.wet_gate_net(inp_cdS))
            delay_scale = 0.15 + 0.85 * torch.sigmoid(self.delay_net(inp_cdS))
            base_gain = F.softplus(self.base_gain_net(torch.cat([context, dec_k, S_prev], dim=-1)))

            infil = infil_coeff * rain_k
            qbase = base_gain * S_prev
            quick_source = wet_gate * (0.20 + S_prev) * rain_k
            drain = drain_coeff * S_prev

            S_next = S_prev + 0.45 * infil - 0.06 * drain - 0.10 * quick_source - 0.05 * qbase
            S_next = F.softplus(S_next)

            S_list.append(S_next.unsqueeze(1))
            qbase_list.append(qbase.unsqueeze(1))
            infil_list.append(infil.unsqueeze(1))
            drain_list.append(drain.unsqueeze(1))
            wet_gate_list.append(wet_gate.unsqueeze(1))
            delay_scale_list.append(delay_scale.unsqueeze(1))
            quick_source_list.append(quick_source.unsqueeze(1))

            S_prev = S_next

        S_seq = torch.cat(S_list, dim=1)                    # [B,P,1]
        qbase_seq = torch.cat(qbase_list, dim=1)            # [B,P,1]
        infil_seq = torch.cat(infil_list, dim=1)
        drain_seq = torch.cat(drain_list, dim=1)
        wet_gate_seq = torch.cat(wet_gate_list, dim=1)      # [B,P,1]
        delay_scale_seq = torch.cat(delay_scale_list, dim=1)# [B,P,1]
        quick_source_seq = torch.cat(quick_source_list, dim=1)

        kernel = self.build_causal_kernel(delay_scale_seq)

        r_eff_delayed = torch.bmm(kernel, quick_source_seq)
        r_eff_seq = 0.65 * quick_source_seq + 0.35 * r_eff_delayed

        vwc_pred = self.vwc_obs_net(
            torch.cat([S_seq, r_eff_seq, qbase_seq, delay_scale_seq], dim=-1)
        )

        return {
            "S_seq": S_seq,
            "qbase_seq": qbase_seq,
            "infil_seq": infil_seq,
            "drain_seq": drain_seq,
            "wet_gate_seq": wet_gate_seq,
            "delay_scale_seq": delay_scale_seq,
            "quick_source_seq": quick_source_seq,
            "kernel": kernel,
            "r_eff_seq": r_eff_seq,
            "vwc_pred": vwc_pred
        }
