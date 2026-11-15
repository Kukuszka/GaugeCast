"""Post-hoc semigroup and phase diagnostics from the GaugeCast++ formulation."""

from __future__ import annotations

from typing import Dict

import torch

from .phase_transport import soft_phase_cdf


def trajectory_defect(short_rollout: torch.Tensor, full_prefix: torch.Tensor) -> torch.Tensor:
    """Trajectory-level prefix semigroup defect."""
    return short_rollout - full_prefix


def phase_defect(
    short_rollout: torch.Tensor,
    full_prefix: torch.Tensor,
    beta: float = 20.0,
) -> torch.Tensor:
    """Event-phase defect under the differentiable cumulative phase map."""
    return soft_phase_cdf(short_rollout, beta=beta) - soft_phase_cdf(full_prefix, beta=beta)


def soft_crest_time(sequence: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    """Differentiable peak-time proxy using softmax mass over lead time."""
    if sequence.ndim == 3:
        sequence = sequence.squeeze(-1)
    P = sequence.shape[1]
    weights = torch.softmax(beta * sequence, dim=1)
    lead = torch.arange(P, device=sequence.device, dtype=sequence.dtype).view(1, P)
    return (weights * lead).sum(dim=1)


def semigroup_phase_report(
    short_rollout: torch.Tensor,
    full_prefix: torch.Tensor,
    beta: float = 20.0,
) -> Dict[str, torch.Tensor]:
    """Diagnostic summary; no gradient update or prediction modification."""
    e = trajectory_defect(short_rollout, full_prefix)
    ep = phase_defect(short_rollout, full_prefix, beta=beta)
    t_short = soft_crest_time(short_rollout, beta=beta)
    t_full = soft_crest_time(full_prefix, beta=beta)
    return {
        "trajectory_defect_l2": torch.linalg.vector_norm(e.flatten(1), dim=1),
        "phase_defect_linf": ep.abs().flatten(1).amax(dim=1),
        "soft_crest_shift_steps": t_short - t_full,
    }
