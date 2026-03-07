"""
ConflictRegularization: penaliza conflito phi-psi.

L_conflict = mean(conflict_intensity^2)

Incentiva operacao alinhada (sem conflito).
"""

import torch
import torch.nn as nn
from typing import Optional


class ConflictRegularization(nn.Module):
    """Regularizacao que penaliza conflito phi-psi.

    Minimiza a intensidade de conflito, incentivando
    alinhamento entre mudancas de phi e quality.
    """

    def forward(
        self,
        conflict_intensity: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computa loss de conflito.

        Args:
            conflict_intensity: [B, T, 1]
            mask: [B, T] (1=valido, 0=padding)

        Returns:
            loss: escalar
        """
        loss_per_token = conflict_intensity.squeeze(-1) ** 2  # [B, T]

        if mask is not None:
            loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss_per_token.mean()

        return loss
