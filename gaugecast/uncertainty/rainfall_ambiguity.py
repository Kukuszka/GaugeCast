from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gaugecast.physics.pde_residual import forward_physics_bundle
from gaugecast.physics.pullback import build_grid


@dataclass
class AmbiguityEvidence:
    """Frozen evidence Z_t used by the inverse ambiguity module."""

    context: torch.Tensor
    rainfall_forecast: torch.Tensor
    predictor_rainfall: torch.Tensor
    base_hydrograph: torch.Tensor
    storage_state: torch.Tensor
    pde_residual: torch.Tensor
    gauge_residual: torch.Tensor
    latent_x: torch.Tensor


@dataclass
class AmbiguityPosteriorOutput:
    """Diagonal Gaussian posterior q_psi(delta r_t | Z_t)."""

    mean: torch.Tensor
    sigma: torch.Tensor
    log_sigma: torch.Tensor


class RainfallAmbiguityPosterior(nn.Module):
    """
    Parameters
    ----------
    feature_dim:
        Number of per-horizon features in ``Z_t`` after feature construction.
    hidden_dim:
        Hidden size of the horizon encoder.
    num_layers:
        Number of recurrent layers.
    min_sigma:
        Positive numerical floor for posterior standard deviation.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        min_sigma: float = 1e-4,
    ) -> None:
        super().__init__()
        self.min_sigma = float(min_sigma)
        self.encoder = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(2 * hidden_dim)
        self.mean_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_sigma_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> AmbiguityPosteriorOutput:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, horizon, feature_dim]")
        z, _ = self.encoder(features)
        z = self.norm(z)
        mean = self.mean_head(z)
        log_sigma = self.log_sigma_head(z).clamp(min=-10.0, max=6.0)
        sigma = F.softplus(log_sigma) + self.min_sigma
        return AmbiguityPosteriorOutput(mean=mean, sigma=sigma, log_sigma=log_sigma)


def freeze_forecaster(model: nn.Module) -> nn.Module:
    """Freeze the base GaugeCast++ forecaster in place and put it in eval mode."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_frozen_evidence(
    model: nn.Module,
    x_hist: torch.Tensor,
    predictor_rainfall: torch.Tensor,
    rainfall_forecast: Optional[torch.Tensor] = None,
) -> AmbiguityEvidence:
    freeze_forecaster(model)
    bundle = forward_physics_bundle(model, x_hist, predictor_rainfall)
    out = bundle["out"]

    r_fc = predictor_rainfall if rainfall_forecast is None else rainfall_forecast
    return AmbiguityEvidence(
        context=out["context"].detach(),
        rainfall_forecast=r_fc.detach(),
        predictor_rainfall=predictor_rainfall.detach(),
        base_hydrograph=out["h_pred"].detach(),
        storage_state=out["storage_out"]["S_seq"].detach(),
        pde_residual=bundle["pde_res"].detach(),
        gauge_residual=bundle["B_gauge"].detach(),
        latent_x=bundle["x"].detach(),
    )


def build_ambiguity_features(evidence: AmbiguityEvidence) -> torch.Tensor:
    """Construct a horizon-wise representation of ``Z_t`` from frozen evidence."""
    B, P, _ = evidence.base_hydrograph.shape
    context = evidence.context.unsqueeze(1).expand(B, P, evidence.context.shape[-1])
    return torch.cat(
        [
            context,
            evidence.rainfall_forecast,
            evidence.predictor_rainfall,
            evidence.base_hydrograph,
            evidence.storage_state,
            evidence.pde_residual.abs(),
            evidence.gauge_residual.abs(),
        ],
        dim=-1,
    )


def rainfall_source_projection(
    discrepancy: torch.Tensor,
    kappa: torch.Tensor | float,
    latent_x: torch.Tensor,
) -> torch.Tensor:
    """Pi_kappa(delta r)_k = delta r_k exp(-kappa x_k)."""
    if not torch.is_tensor(kappa):
        kappa = torch.as_tensor(kappa, dtype=discrepancy.dtype, device=discrepancy.device)
    return discrepancy * torch.exp(-kappa * latent_x)


def projected_physics_loss(
    posterior_mean: torch.Tensor,
    frozen_residual: torch.Tensor,
    kappa: torch.Tensor | float,
    latent_x: torch.Tensor,
) -> torch.Tensor:
    """Equation-(74)-style projected residual matching loss."""
    projected = rainfall_source_projection(posterior_mean, kappa, latent_x)
    return F.mse_loss(projected, frozen_residual)


def heteroscedastic_nll(
    observed_discrepancy: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_sigma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Diagonal-Gaussian negative log likelihood from Appendix F."""
    var = posterior_sigma.square() + eps
    return 0.5 * (torch.log(var) + (observed_discrepancy - posterior_mean).square() / var).mean()


def corrected_rainfall_bands(
    rainfall_forecast: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_sigma: torch.Tensor,
    z_value: float = 1.645,
) -> Dict[str, torch.Tensor]:
    """Posterior mean rainfall and symmetric pointwise credibility bands."""
    corrected_mean = rainfall_forecast + posterior_mean
    return {
        "mean": corrected_mean,
        "lower": corrected_mean - z_value * posterior_sigma,
        "upper": corrected_mean + z_value * posterior_sigma,
    }


def rainfall_to_stage_jacobian_autograd(
    model: nn.Module,
    x_hist: torch.Tensor,
    predictor_rainfall: torch.Tensor,
) -> torch.Tensor:
    """Compute J_t = d h_hat / d r using the frozen forecaster.

    Returns a tensor with shape ``[B, P_out, P_rain]``.  Model parameters are
    frozen, but gradients with respect to the rainfall input remain enabled.
    """
    freeze_forecaster(model)
    rainfall = predictor_rainfall.detach().clone().requires_grad_(True)
    B, P, _ = rainfall.shape
    tau, x = build_grid(B, P, rainfall.device, requires_grad=False)
    pred = model(x_hist.detach(), rainfall, tau, x)["h_pred"].squeeze(-1)

    rows = []
    for j in range(P):
        grad_j = torch.autograd.grad(
            pred[:, j].sum(),
            rainfall,
            retain_graph=(j < P - 1),
            create_graph=False,
            allow_unused=False,
        )[0].squeeze(-1)
        rows.append(grad_j)
    return torch.stack(rows, dim=1).detach()


@torch.no_grad()
def rainfall_to_stage_jacobian_finite_difference(
    model: nn.Module,
    x_hist: torch.Tensor,
    predictor_rainfall: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Central finite-difference estimator corresponding to Eq. (85)."""
    freeze_forecaster(model)
    B, P, _ = predictor_rainfall.shape
    tau, x = build_grid(B, P, predictor_rainfall.device, requires_grad=False)
    cols = []
    for k in range(P):
        plus = predictor_rainfall.clone()
        minus = predictor_rainfall.clone()
        plus[:, k, 0] += eps
        minus[:, k, 0] -= eps
        hp = model(x_hist, plus, tau, x)["h_pred"].squeeze(-1)
        hm = model(x_hist, minus, tau, x)["h_pred"].squeeze(-1)
        cols.append((hp - hm) / (2.0 * eps))
    # each list item is [B, P_out] for one rainfall coordinate k
    return torch.stack(cols, dim=-1)  # [B, P_out, P_rain]


def sensitivity_energy(jacobian: torch.Tensor) -> torch.Tensor:
    """s^2_{t,k} = ||J_{t,:,k}||_2^2, shape [B, P]."""
    if jacobian.ndim != 3:
        raise ValueError("jacobian must have shape [batch, output_horizon, rain_horizon]")
    return jacobian.square().sum(dim=1)


def ambiguity_field(
    posterior_sigma: torch.Tensor,
    jacobian: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return horizon-wise ambiguity a_{t,k} and issue-level score A_t."""
    sigma2 = posterior_sigma.squeeze(-1).square()
    s2 = sensitivity_energy(jacobian)
    field = sigma2 * s2
    score = field.mean(dim=1)
    return field, score


def ambiguity_trace_score(posterior_sigma: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
    """Equivalent trace-form implementation P^-1 tr(J Sigma J^T)."""
    _, score = ambiguity_field(posterior_sigma, jacobian)
    return score


def normalize_nonnegative_sequence(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Nonnegative sequence normalization used for ambiguity/error alignment."""
    z = z - z.amin(dim=-1, keepdim=True)
    return z / (z.sum(dim=-1, keepdim=True) + eps)


def hydrograph_error_field(base_hydrograph: torch.Tensor, observed_stage: torch.Tensor) -> torch.Tensor:
    """Squared hydrograph error e_{t,k}."""
    if base_hydrograph.ndim == 3:
        base_hydrograph = base_hydrograph.squeeze(-1)
    if observed_stage.ndim == 3:
        observed_stage = observed_stage.squeeze(-1)
    return (base_hydrograph - observed_stage).square()


def ambiguity_error_alignment(
    ambiguity: torch.Tensor,
    hydrograph_error: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    a_bar = normalize_nonnegative_sequence(ambiguity)
    e_bar = normalize_nonnegative_sequence(hydrograph_error)
    d = (a_bar - e_bar).square()
    if reduction == "none":
        return d
    if reduction == "sum":
        return d.sum()
    return d.mean()


def ambiguity_objective(
    posterior: AmbiguityPosteriorOutput,
    evidence: AmbiguityEvidence,
    kappa: torch.Tensor | float,
    observed_rainfall: Optional[torch.Tensor] = None,
    lambda_phys: float = 1.0,
    lambda_nll: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Train the inverse module without modifying the base forecaster.

    If observed rainfall is supplied, the discrepancy target is
    ``r_obs - r_fc`` and the heteroscedastic NLL is included.  Otherwise only
    the frozen residual projection term is used.
    """
    phys = projected_physics_loss(
        posterior.mean,
        evidence.pde_residual,
        kappa,
        evidence.latent_x,
    )
    nll = torch.zeros((), dtype=phys.dtype, device=phys.device)
    if observed_rainfall is not None:
        delta_obs = observed_rainfall - evidence.rainfall_forecast
        nll = heteroscedastic_nll(delta_obs, posterior.mean, posterior.sigma)
    total = lambda_phys * phys + lambda_nll * nll
    return {"total": total, "physics": phys, "nll": nll}


def assert_strict_decoupling(
    base_prediction_before: torch.Tensor,
    base_prediction_after: torch.Tensor,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """Raise if a post-hoc ambiguity computation changed the base forecast."""
    if not torch.allclose(base_prediction_before, base_prediction_after, atol=atol, rtol=rtol):
        raise AssertionError("Ambiguity diagnostics changed the deterministic base hydrograph.")
