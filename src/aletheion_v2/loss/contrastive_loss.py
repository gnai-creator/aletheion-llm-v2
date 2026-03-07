"""
ContrastiveRegularization: regulariza divergencia metacognitiva.

L_contrastive = -mean(log(divergence + eps)) + cap * mean(relu(divergence - cap_threshold)^2)

Incentiva divergencia informativa mas penaliza divergencia excessiva.
"""

import torch
import torch.nn as nn
from typing import Optional


class ContrastiveRegularization(nn.Module):
    """Regularizacao da divergencia contrastiva.

    Dois termos:
    1. -log(divergence): incentiva divergencia > 0 (evita colapso)
    2. relu(divergence - cap)^2: penaliza divergencia > cap (evita explosao)

    Args:
        cap_threshold: threshold de capping (default 0.8)
        cap_weight: peso do termo de capping (default 0.1)
        eps: epsilon para log (default 1e-8)
    """

    def __init__(
        self,
        cap_threshold: float = 0.8,
        cap_weight: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.cap_threshold = cap_threshold
        self.cap_weight = cap_weight
        self.eps = eps

    def forward(
        self,
        divergence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computa loss contrastiva.

        Args:
            divergence: [B, T, 1]
            mask: [B, T]

        Returns:
            loss: escalar
        """
        div = divergence.squeeze(-1)  # [B, T]

        # Termo 1: evita colapso
        anti_collapse = -torch.log(div + self.eps)

        # Termo 2: evita explosao
        excess = torch.relu(div - self.cap_threshold)
        anti_explosion = excess ** 2

        loss_per_token = anti_collapse + self.cap_weight * anti_explosion

        if mask is not None:
            loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss_per_token.mean()

        return loss
