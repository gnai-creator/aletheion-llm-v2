"""
MAD Calibration Loss: Aprendizado de tau^2.

Forca tau^2 a se calibrar para que:
- Tokens corretos tenham alta confianca (d^2 baixo)
- Tokens incorretos tenham baixa confianca (d^2 alto)

Formula:
    confidence = exp(-0.5 * d_sq)
    correct = (argmax(logits) == labels).float()
    L_mad = BCE(confidence, correct)

Isso calibra tau^2 via backprop, emulando update bayesiano.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MADCalibrationLoss(nn.Module):
    """Loss para calibrar tau^2 do MAD.

    Usa Binary Cross-Entropy entre confianca MAD e acuracia real.
    Gradientes fluem ate log_tau_sq no BayesianTau.

    Args:
        eps: Epsilon para clamp de confianca
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        confidence: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Computa MAD calibration loss.

        Args:
            confidence: [B, T, 1] confianca MAD
            logits: [B, T, V] logits do modelo
            labels: [B, T] labels reais
            mask: [B, T] mascara (1 = valido)

        Returns:
            loss: escalar
        """
        # Target: token correto?
        predictions = logits.argmax(dim=-1)  # [B, T]
        correct = (predictions == labels).float()  # [B, T]

        # BCE entre confianca e acuracia
        # Usa binary_cross_entropy_with_logits (safe para autocast bf16)
        conf = confidence.squeeze(-1).clamp(self.eps, 1.0 - self.eps)
        conf_logits = torch.log(conf / (1.0 - conf))  # inverse sigmoid -> logits
        loss = F.binary_cross_entropy_with_logits(
            conf_logits, correct, reduction="none"
        )  # [B, T]

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)

        return loss.mean()
