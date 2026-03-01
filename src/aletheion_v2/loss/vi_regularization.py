"""
VI Regularization: Penaliza phi(M) baixo.

Incentiva o modelo a manter o manifold saudavel (phi > phi_critical).
A penalidade cresce quadraticamente quando phi cai abaixo do limiar.

Formula:
    deficit = max(0, phi_critical - phi_total)
    L_vi = mean(deficit^2)

Tambem penaliza severity alta (VI nao deve ser ativado frequentemente).
"""

import torch
import torch.nn as nn


class VIRegularization(nn.Module):
    """Regularizacao baseada em saude do manifold (phi).

    Dois componentes:
    1. phi_penalty: penaliza phi < phi_critical
    2. severity_penalty: penaliza severity alta (uso excessivo do VI)

    Args:
        phi_critical: Limiar critico de phi
        severity_weight: Peso da penalidade de severity
    """

    def __init__(
        self,
        phi_critical: float = 0.5,
        severity_weight: float = 0.3,
    ):
        super().__init__()
        self.phi_critical = phi_critical
        self.severity_weight = severity_weight

    def forward(
        self,
        phi_total: torch.Tensor,
        vi_severity: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Computa VI regularization loss.

        Args:
            phi_total: [B, T, 1] saude do manifold
            vi_severity: [B, T, 1] severidade do VI
            mask: [B, T] mascara (1 = valido)

        Returns:
            loss: escalar
        """
        # Penalidade por phi baixo
        deficit = (self.phi_critical - phi_total.squeeze(-1)).clamp(min=0.0)
        phi_loss = deficit ** 2  # [B, T]

        # Penalidade por severity alta
        sev_loss = vi_severity.squeeze(-1) ** 2  # [B, T]

        # Combina
        loss = phi_loss + self.severity_weight * sev_loss

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)

        return loss.mean()
