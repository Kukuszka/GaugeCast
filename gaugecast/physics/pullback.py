import torch
import torch.autograd as autograd


def grad(outputs, inputs):
    return autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

def build_grid(batch_size, pred_len, device, requires_grad=False):
    tau = torch.linspace(0.0, 1.0, pred_len, device=device).view(1, pred_len, 1).repeat(batch_size, 1, 1)
    x   = torch.linspace(0.0, 1.0, pred_len, device=device).view(1, pred_len, 1).repeat(batch_size, 1, 1)

    if requires_grad:
        tau.requires_grad_(True)
        x.requires_grad_(True)
    return tau, x

def compute_pullback_terms(model_out, tau, x):
    phi = model_out["phi"]
    v   = model_out["v"]

    phi_tau = grad(phi, tau)
    phi_x   = grad(phi, x)
    phi_xx  = grad(phi_x, x)

    v_tau = grad(v, tau)
    v_y   = grad(v, phi)
    v_yy  = grad(v_y, phi)

    return {
        "phi_tau": phi_tau,
        "phi_x": phi_x,
        "phi_xx": phi_xx,
        "v_tau": v_tau,
        "v_y": v_y,
        "v_yy": v_yy
    }

def compute_diffeomorphism_loss(phi, x, phi_x, eps=1e-6):
    barrier = torch.relu(eps - phi_x).mean()
    identity_reg = ((phi - x) ** 2).mean()

    log_jac = torch.log(phi_x.clamp_min(eps))
    smooth_log_jac = ((log_jac[:, 1:, :] - log_jac[:, :-1, :]) ** 2).mean()

    return barrier + 0.1 * identity_reg + 0.1 * smooth_log_jac
