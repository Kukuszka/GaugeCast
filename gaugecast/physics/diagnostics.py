"""Paper-aligned pullback diagnostics that do not affect model training."""

from __future__ import annotations

from typing import Dict

import torch


def effective_latent_diffusion(nu: torch.Tensor, phi_x: torch.Tensor) -> torch.Tensor:
    """Effective diffusion coefficient nu * (d_x phi)^2."""
    return nu * phi_x.square()


def transformed_peclet_number(
    gauge_residual: torch.Tensor,
    nu: torch.Tensor,
    phi_x: torch.Tensor,
    length_scale: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pe_psi = |B_psi| ell / (nu (d_x phi)^2)."""
    denom = nu * phi_x.square()
    return gauge_residual.abs() * float(length_scale) / denom.clamp_min(eps)


def identity_gauge_residual(advection: torch.Tensor) -> torch.Tensor:
    """Residual transport for the identity coordinate map phi(x,t)=x."""
    return advection


def residual_transport_ratio(
    learned_gauge_residual: torch.Tensor,
    identity_residual: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pointwise |B_psi| / |B_identity| diagnostic."""
    return learned_gauge_residual.abs() / identity_residual.abs().clamp_min(eps)


def pullback_diagnostic_report(
    gauge_residual: torch.Tensor,
    advection: torch.Tensor,
    nu: torch.Tensor,
    phi_x: torch.Tensor,
    length_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Aggregate paper-style transport/stiffness diagnostics."""
    identity = identity_gauge_residual(advection)
    ratio = residual_transport_ratio(gauge_residual, identity)
    pe = transformed_peclet_number(gauge_residual, nu, phi_x, length_scale=length_scale)
    diff = effective_latent_diffusion(nu, phi_x)

    return {
        "mean_abs_gauge_residual": gauge_residual.abs().mean(),
        "median_residual_ratio": ratio.median(),
        "transport_suppression_fraction": (gauge_residual.abs() < identity.abs()).float().mean(),
        "mean_transformed_peclet": pe.mean(),
        "max_transformed_peclet": pe.max(),
        "mean_effective_diffusion": diff.mean(),
        "min_phi_x": phi_x.min(),
    }
