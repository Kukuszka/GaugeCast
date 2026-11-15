import torch
import torch.nn.functional as F

from gaugecast.physics.pullback import build_grid
from .phase_transport import soft_phase_cdf


def run_model_rollout(model, x_hist_full, fut_seq):
    B = x_hist_full.size(0)
    P = fut_seq.size(1)
    tau, x = build_grid(B, P, x_hist_full.device, requires_grad=False)
    return model(x_hist_full, fut_seq, tau, x)

def compute_semigroup_phase_losses(model, x_hist, fut_seq, pred_full, split, beta=20.0):
    """
    Leakage-free version:
    direct short rollout from original history should match
    the prefix of the full rollout.
    """
    pred_full_head = pred_full[:, :split, :].detach()

    out_short = run_model_rollout(model, x_hist, fut_seq[:, :split, :])
    pred_short = out_short["h_pred"]

    sg_prefix = F.mse_loss(pred_short, pred_full_head)

    ph_prefix = F.mse_loss(
        soft_phase_cdf(pred_short, beta=beta),
        soft_phase_cdf(pred_full_head, beta=beta)
    )

    # no tail continuation because that would require unavailable future state
    sg_tail = torch.tensor(0.0, device=x_hist.device)
    ph_tail = torch.tensor(0.0, device=x_hist.device)

    return {
        "sg_prefix": sg_prefix,
        "sg_tail": sg_tail,
        "ph_prefix": ph_prefix,
        "ph_tail": ph_tail
    }
