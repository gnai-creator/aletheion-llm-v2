"""
STP Loss: Smooth Transition Penalty.

Encourages smooth transitions in hidden state space by penalizing
non-linear trajectories between token positions.

Reference:
    Guo et al. (2025) - "Straight-Through Meets Sparse Recovery"
    arXiv:2602.22617v1

For random triplets (s, r, t) with s < r < t, the loss is:
    L_stp = 1 - cos_sim(h_t - h_r, h_r - h_s)

This encourages the hidden state trajectory to be locally linear,
which improves representation smoothness and training stability.
"""

import random

import torch
import torch.nn.functional as F


def stp_loss(
    hidden_states: torch.Tensor,
    num_triplets: int = 1,
) -> torch.Tensor:
    """Compute STP loss over random triplets.

    Args:
        hidden_states: [B, T, D] hidden states from transformer
        num_triplets: number of random triplets to sample

    Returns:
        loss: scalar, mean STP loss over triplets and batch
    """
    seq_len = hidden_states.shape[1]
    if seq_len < 3:
        return torch.tensor(0.0, device=hidden_states.device)

    total = torch.tensor(0.0, device=hidden_states.device)
    for _ in range(num_triplets):
        s, r, t = sorted(random.sample(range(seq_len), 3))
        hs = hidden_states[:, s]  # [B, D]
        hr = hidden_states[:, r]  # [B, D]
        ht = hidden_states[:, t]  # [B, D]
        d1 = ht - hr
        d2 = hr - hs
        # Skip triplet if either difference vector has near-zero norm —
        # cosine_similarity divides by norms, producing NaN gradients
        # when norms approach zero.
        n1 = d1.norm(dim=-1, keepdim=True)
        n2 = d2.norm(dim=-1, keepdim=True)
        if (n1 < 1e-6).any() or (n2 < 1e-6).any():
            continue
        cos = F.cosine_similarity(d1, d2, dim=-1)  # [B]
        total = total + (1.0 - cos.mean())

    return total / num_triplets
