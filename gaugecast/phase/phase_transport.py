import torch


def soft_phase_cdf(seq, beta=20.0):
    """
    seq: [B, P, 1]
    returns: [B, P]
    """
    w = torch.softmax(beta * seq.squeeze(-1), dim=1)
    return torch.cumsum(w, dim=1)
