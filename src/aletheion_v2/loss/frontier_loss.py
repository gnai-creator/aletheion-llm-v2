"""
FrontierRegularization: incentiva exploracao de fronteiras.

L_frontier = -mean(frontier_score * log(novelty + eps))

Maximiza frontier_score em regioes de alta novidade.
"""

import torch
import torch.nn as nn
from typing import Optional


class FrontierRegularization(nn.Module):
    """Regularizacao que incentiva exploracao de fronteiras.

    Usa log-novelty ponderado pelo frontier_score como sinal
    de recompensa para exploracao.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        frontier_score: torch.Tensor,
        novelty: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computa loss de exploracao.

        Args:
            frontier_score: [B, T, 1]
            novelty: [B, T, 1]
            mask: [B, T]

        Returns:
            loss: escalar (negativo para maximizar)
        """
        # Negativo: queremos MAXIMIZAR exploracao
        log_novelty = torch.log(novelty.squeeze(-1) + self.eps)
        loss_per_token = -(frontier_score.squeeze(-1) * log_novelty)

        if mask is not None:
            loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss_per_token.mean()

        return loss
