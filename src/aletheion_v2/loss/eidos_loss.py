"""
EidosRegularization: penaliza desbalanceamento de eixos.

L_eidos = mean((axis_balance.std(dim=-1) - target_std)^2)

Incentiva distribuicao uniforme de eixos no manifold 5D.
"""

import torch
import torch.nn as nn
from typing import Optional


class EidosRegularization(nn.Module):
    """Regularizacao de balanceamento de eixos.

    Penaliza quando o std do axis_balance se desvia do target.
    Target 0.15 permite variacao natural sem desbalanceamento.

    Args:
        target_std: desvio padrao alvo do balance (default 0.15)
    """

    def __init__(self, target_std: float = 0.15):
        super().__init__()
        self.target_std = target_std

    def forward(
        self,
        axis_balance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computa loss de regularizacao.

        Args:
            axis_balance: [B, T, 5] balanceamento por eixo
            mask: [B, T] mascara (1=valido, 0=padding)

        Returns:
            loss: escalar
        """
        # std across eixos por token: [B, T]
        balance_std = axis_balance.std(dim=-1)

        # Penaliza desvio do target
        loss_per_token = (balance_std - self.target_std) ** 2

        if mask is not None:
            loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss_per_token.mean()

        return loss
