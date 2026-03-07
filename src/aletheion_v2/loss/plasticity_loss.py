"""
PlasticityRegularization: impede deplecao total da plasticidade.

L_plasticity = mean(relu(min_plasticity - plasticity)^2)
"""

import torch
import torch.nn as nn
from typing import Optional


class PlasticityRegularization(nn.Module):
    """Regularizacao de plasticidade.

    Penaliza quando plasticidade cai abaixo do minimo.

    Args:
        min_plasticity: threshold minimo (default 0.3)
    """

    def __init__(self, min_plasticity: float = 0.3):
        super().__init__()
        self.min_plasticity = min_plasticity

    def forward(
        self,
        plasticity: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computa loss de plasticidade.

        Args:
            plasticity: [B, T, 1]
            mask: [B, T]

        Returns:
            loss: escalar
        """
        shortfall = torch.relu(
            self.min_plasticity - plasticity.squeeze(-1)
        )
        loss_per_token = shortfall ** 2

        if mask is not None:
            loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss_per_token.mean()

        return loss
