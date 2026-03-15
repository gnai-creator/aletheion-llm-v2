"""
MAD Confidence: C(p) = exp(-d^2 / 2*tau^2).

Confianca baseada em distancia ao truth centroid.
Usa Gaussian decay para mapear distancia em confianca [0, 1].

Tres modos:
- G=None: diagonal Mahalanobis via BayesianTau
- G [D,D]: Mahalanobis completa constante
- G [B,T,D,D]: Mahalanobis com tensor metrico local (curvatura real)
"""

import torch
import torch.nn as nn
from typing import Optional

from aletheion_v2.mad.bayesian_tau import BayesianTau


class MADConfidence(nn.Module):
    """Confianca MAD via Gaussian decay.

    Combina:
    1. BayesianTau para tau^2 aprendivel (escala)
    2. Gaussian kernel para mapear distancia -> confianca

    Args:
        drm_dim: Dimensao do manifold
        per_axis: tau por eixo
        init_log_tau_sq: log(tau^2) inicial
    """

    def __init__(
        self,
        drm_dim: int = 5,
        per_axis: bool = True,
        init_log_tau_sq: float = 0.0,
    ):
        super().__init__()
        self.drm_dim = drm_dim
        self.tau = BayesianTau(drm_dim, per_axis, init_log_tau_sq)

    def forward(
        self,
        coords: torch.Tensor,
        truth_centroid: torch.Tensor,
        G: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computa confianca MAD.

        Args:
            coords: [B, T, drm_dim] coordenadas no manifold
            truth_centroid: [drm_dim] centroide truth
            G: tensor metrico - [D,D] constante ou [B,T,D,D] local

        Returns:
            confidence: [B, T, 1] confianca em [0, 1]
            distance_sq: [B, T, 1] distancia^2
            tau_sq: [drm_dim] ou escalar (para loss de calibracao)
        """
        tau_sq = self.tau.get_tau_sq()

        if G is not None:
            delta = coords - truth_centroid.unsqueeze(0).unsqueeze(0)

            if G.dim() == 4:
                # G(x) local: [B, T, D, D]
                Gd = torch.matmul(
                    delta.unsqueeze(-2), G
                ).squeeze(-2)  # [B, T, D]
            else:
                # G constante: [D, D]
                Gd = torch.matmul(delta, G)  # [B, T, D]

            d_sq_metric = (Gd * delta).sum(dim=-1, keepdim=True)
            d_sq_metric = d_sq_metric.clamp(min=0.0)

            # Normalizar por tau medio para manter escala compativel
            avg_tau_sq = tau_sq.mean() if self.tau.per_axis else tau_sq
            d_sq = d_sq_metric / avg_tau_sq.clamp(min=1e-6)
        else:
            # Fallback: diagonal Mahalanobis via BayesianTau
            d_sq, tau_sq = self.tau(coords, truth_centroid)

        # Gaussian decay: C = exp(-d^2 / 2)
        confidence = torch.exp(-0.5 * d_sq)

        return confidence, d_sq, tau_sq
