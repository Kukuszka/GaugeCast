import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import HistoryEncoder
from .decoder import FutureRainDecoder, LatentSolutionNet
from .event_residual import EventResidualHead
from .storage import StorageDynamicsModule
from gaugecast.physics.gauge_operator import GaugeNet


class GaugeHydroDecoderPINN(nn.Module):
    """
    Contribution 1 backbone + storage-conditioned forcing extension.
    """
    def __init__(self, hist_input_dim, hist_input_idx, rain_idx, vwc_idx, hidden_dim=64):
        super().__init__()

        self.hist_input_idx = hist_input_idx
        self.rain_idx = rain_idx
        self.vwc_idx = vwc_idx

        self.encoder = HistoryEncoder(
            input_dim=hist_input_dim,
            hidden_dim=hidden_dim
        )

        self.decoder = FutureRainDecoder(
            input_dim=1,
            hidden_dim=hidden_dim
        )

        self.storage_module = StorageDynamicsModule(
            context_dim=hidden_dim,
            dec_dim=hidden_dim,
            hidden_dim=hidden_dim
        )

        hydro_cond_dim = hidden_dim + 3  # dec_feat + S + r_eff + delay_scale

        self.gauge_net = GaugeNet(
            context_dim=hidden_dim,
            dec_dim=hydro_cond_dim,
            hidden_dim=hidden_dim
        )

        self.solution_net = LatentSolutionNet(
            context_dim=hidden_dim,
            dec_dim=hydro_cond_dim,
            hidden_dim=hidden_dim
        )

        self.event_head = EventResidualHead(
            context_dim=hidden_dim,
            dec_dim=hydro_cond_dim,
            hidden_dim=hidden_dim
        )

        self.a_head = nn.Sequential(
            nn.Linear(hydro_cond_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.log_nu = nn.Parameter(torch.tensor(-2.0))
        self.log_kappa = nn.Parameter(torch.tensor(0.0))
        self.log_alpha_q = nn.Parameter(torch.tensor(-1.0))
        self.log_alpha_b = nn.Parameter(torch.tensor(-1.5))

    def compute_forcing(self, fut_seq, x, r_eff, S_seq, qbase):
        kappa = F.softplus(self.log_kappa) + 1e-6
        beta_r = F.softplus(self.log_alpha_q) + 1e-6
        beta_s = F.softplus(self.log_alpha_b) + 1e-6

        rain_direct = 0.60 * fut_seq * torch.exp(-kappa * x)
        runoff_term = 0.35 * beta_r * r_eff * torch.exp(-kappa * x)
        storage_term = 0.08 * beta_s * S_seq
        base_term = 0.08 * qbase

        forcing = rain_direct + runoff_term + storage_term + base_term
        return forcing

    def compute_a(self, hydro_cond, tau, x):
        inp = torch.cat([hydro_cond, tau, x], dim=-1)
        a = self.a_head(inp)
        return a

    def forward(self, x_hist_full, fut_seq, tau, x):
        x_hist_enc = x_hist_full[:, :, self.hist_input_idx]
        context = self.encoder(x_hist_enc)
        dec_feat = self.decoder(fut_seq, context)

        last_rain = x_hist_full[:, -1, self.rain_idx:self.rain_idx+1]
        last_vwc  = x_hist_full[:, -1, self.vwc_idx:self.vwc_idx+1]

        storage_out = self.storage_module(
            last_vwc=last_vwc,
            last_rain=last_rain,
            fut_seq=fut_seq,
            context=context,
            dec_feat=dec_feat
        )

        S_seq = storage_out["S_seq"]
        r_eff_seq = storage_out["r_eff_seq"]
        delay_scale_seq = storage_out["delay_scale_seq"]
        qbase_seq = storage_out["qbase_seq"]

        hydro_cond = torch.cat([dec_feat, S_seq, r_eff_seq, delay_scale_seq], dim=-1)

        phi = self.gauge_net(tau, x, context, hydro_cond)
        v = self.solution_net(phi, tau, context, hydro_cond)

        delta_event, event_residual, event_gate = self.event_head(
            tau=tau,
            x=x,
            fut_seq=fut_seq,
            context=context,
            dec_feat=hydro_cond
        )

        h_pred = v + event_residual

        a = self.compute_a(hydro_cond, tau, x)
        nu = F.softplus(self.log_nu) + 1e-6

        forcing = self.compute_forcing(
            fut_seq=fut_seq,
            x=x,
            r_eff=r_eff_seq,
            S_seq=S_seq,
            qbase=qbase_seq
        )

        return {
            "context": context,
            "dec_feat": dec_feat,
            "storage_out": storage_out,
            "phi": phi,
            "v": v,
            "delta_event": delta_event,
            "event_residual": event_residual,
            "event_gate": event_gate,
            "h_pred": h_pred,
            "a": a,
            "r_eff_seq": r_eff_seq,
            "delay_scale_seq": delay_scale_seq,
            "nu": nu,
            "forcing": forcing
        }
