"""
VARO Loss: Calibracao de incerteza Q1/Q2.

Penaliza discrepancia entre confianca predita (Q1*Q2) e
acuracia real (token correto ou nao).

Formula:
    c = q1 * q2
    correct = (argmax(logits) == labels).float()
    L_varo = MSE(c, correct)

Isso forca Q1*Q2 a refletir a probabilidade real de acerto.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAROLoss(nn.Module):
    """Variational Reliability Optimization loss.

    Calibra Q1*Q2 para refletir probabilidade de acerto.

    Args:
        eps: Epsilon para estabilidade
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Computa VARO loss.

        Args:
            q1: [B, T, 1] incerteza aleatoria
            q2: [B, T, 1] incerteza epistemica
            logits: [B, T, V] logits do modelo
            labels: [B, T] labels reais
            mask: [B, T] mascara (1 = valido, 0 = padding)

        Returns:
            loss: escalar
        """
        # Confianca predita
        confidence = (q1 * q2).squeeze(-1)  # [B, T]

        # Acuracia real (soft: probabilidade do token correto)
        probs = F.softmax(logits, dim=-1)  # [B, T, V]
        correct_probs = probs.gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)  # [B, T]

        # MSE entre confianca e acuracia real
        loss = (confidence - correct_probs) ** 2  # [B, T]

        # Aplica mascara
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + self.eps)

        return loss.mean()
