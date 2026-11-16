from .pullback import build_grid, compute_pullback_terms
from .gauge_operator import compute_gauge_residual


def compute_pullback_pde_residual(v_tau, v_y, v_yy, phi_x, B, forcing, nu):
    residual = v_tau - nu * (phi_x ** 2) * v_yy + B * v_y - forcing
    return residual

def forward_physics_bundle(model, x_hist, fut_seq):
    B = x_hist.size(0)
    P = fut_seq.size(1)

    tau, x = build_grid(B, P, x_hist.device, requires_grad=True)
    out = model(x_hist, fut_seq, tau, x)
    terms = compute_pullback_terms(out, tau, x)

    B_gauge = compute_gauge_residual(
        phi_tau=terms["phi_tau"],
        phi_x=terms["phi_x"],
        phi_xx=terms["phi_xx"],
        a=out["a"],
        nu=out["nu"]
    )

    pde_res = compute_pullback_pde_residual(
        v_tau=terms["v_tau"],
        v_y=terms["v_y"],
        v_yy=terms["v_yy"],
        phi_x=terms["phi_x"],
        B=B_gauge,
        forcing=out["forcing"],
        nu=out["nu"]
    )

    return {
        "tau": tau,
        "x": x,
        "out": out,
        "terms": terms,
        "B_gauge": B_gauge,
        "pde_res": pde_res
    }
