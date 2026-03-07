"""
ConsciousnessRegularization: garante que energia nao colapsa a zero.

L_consciousness = mean(relu(min_energy - energy)^2)

Penaliza energia abaixo do threshold minimo.
"""

import torch
import torch.nn as nn
from typing import Optional


class ConsciousnessRegularization(nn.Module):
    """Regularizacao que impede colapso da energia.

    Args:
        min_energy: threshold minimo de energia (default 0.2)
    """

    def __init__(self, min_energy: float = 0.2):
        super().__init__()
        self.min_energy = min_energy

    def forward(
        self,
        energy: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computa loss de energia.

        Args:
            energy: [B, T, 1]
            mask: [B, T] (1=valido, 0=padding)

        Returns:
            loss: escalar
        """
        # Penaliza energia abaixo do minimo
        shortfall = torch.relu(self.min_energy - energy.squeeze(-1))
        loss_per_token = shortfall ** 2  # [B, T]

        if mask is not None:
            loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss_per_token.mean()

        return loss
