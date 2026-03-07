"""
GroundingRegularization: entropia + calibracao da ambiguidade.

L_grounding = lambda_entropy * entropy_reg + lambda_ambiguity * MSE(ambiguity, q1)

Entropia: incentiva classificacao decisiva (low entropy em task_probs).
Calibracao: ambiguidade deve alinhar com incerteza aleatoria (q1).
"""

import torch
import torch.nn as nn
from typing import Optional


class GroundingRegularization(nn.Module):
    """Regularizacao de grounding.

    Combina entropia de classificacao de tarefa com
    calibracao da ambiguidade contra Q1.

    Args:
        entropy_weight: peso da entropia (default 0.5)
        calibration_weight: peso da calibracao (default 0.5)
    """

    def __init__(
        self,
        entropy_weight: float = 0.5,
        calibration_weight: float = 0.5,
    ):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.calibration_weight = calibration_weight

    def forward(
        self,
        task_probs: torch.Tensor,
        ambiguity_level: torch.Tensor,
        q1: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computa loss de grounding.

        Args:
            task_probs: [B, T, 9] probabilidades de tarefa
            ambiguity_level: [B, T, 1] nivel de ambiguidade
            q1: [B, T, 1] incerteza aleatoria
            mask: [B, T] mascara

        Returns:
            loss: escalar
        """
        # Entropia de classificacao: incentiva decisividade
        # H = -sum(p * log(p + eps))
        entropy = -(task_probs * torch.log(task_probs + 1e-8)).sum(dim=-1)
        # [B, T]

        # Calibracao: ambiguidade deve alinhar com q1
        calibration = (ambiguity_level.squeeze(-1) - q1.squeeze(-1)) ** 2
        # [B, T]

        loss_per_token = (
            self.entropy_weight * entropy
            + self.calibration_weight * calibration
        )

        if mask is not None:
            loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss_per_token.mean()

        return loss
