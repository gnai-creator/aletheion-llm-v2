"""
MOPsiRegularization: alinha psi com confianca e phi.

L_mopsi = mean((psi - confidence)^2) * (1 - conflict)

Quando conflito e baixo, psi deve alinhar com confianca.
Quando conflito e alto, a penalidade diminui.
"""

import torch
import torch.nn as nn
from typing import Optional


class MOPsiRegularization(nn.Module):
    """Regularizacao que alinha psi com confianca.

    Psi (satisfacao) deve correlacionar com confianca
    na ausencia de conflito.
    """

    def forward(
        self,
        psi: torch.Tensor,
        confidence: torch.Tensor,
        conflict_intensity: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computa loss de alinhamento psi-confianca.

        Args:
            psi: [B, T, 1]
            confidence: [B, T, 1]
            conflict_intensity: [B, T, 1]
            mask: [B, T]

        Returns:
            loss: escalar
        """
        # MSE ponderado por (1 - conflito)
        mse = (psi.squeeze(-1) - confidence.squeeze(-1)) ** 2
        weight = (1.0 - conflict_intensity.squeeze(-1)).clamp(0, 1)
        loss_per_token = mse * weight

        if mask is not None:
            loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss_per_token.mean()

        return loss
