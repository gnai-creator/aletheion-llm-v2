"""
MAD Confidence: C(p) = exp(-d^2 / 2*tau^2).

Confianca baseada em distancia de Mahalanobis ao truth centroid.
Usa Gaussian decay para mapear distancia em confianca [0, 1].

Formulas:
    C(p) = exp(-d^2 / (2 * tau^2))         -- isotropico
    C(p) = exp(-0.5 * sum(delta_i^2/tau_i^2))  -- anisotropico

Propriedades:
    - C = 1 quando p = truth (d = 0)
    - C -> 0 quando p esta longe do truth
    - tau controla velocidade de decaimento
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from aletheion_v2.mad.bayesian_tau import BayesianTau


class MADConfidence(nn.Module):
    """Confianca MAD via Gaussian decay.

    Combina:
    1. BayesianTau para tau^2 aprendivel
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computa confianca MAD.

        Args:
            coords: [B, T, drm_dim] coordenadas no manifold
            truth_centroid: [drm_dim] centroide truth

        Returns:
            confidence: [B, T, 1] confianca em [0, 1]
            distance_sq: [B, T, 1] distancia^2 Mahalanobis
            tau_sq: [drm_dim] ou escalar (para loss de calibracao)
        """
        # Distancia de Mahalanobis ao quadrado
        d_sq, tau_sq = self.tau(coords, truth_centroid)

        # Gaussian decay: C = exp(-d^2 / 2)
        # Nota: d_sq ja inclui divisao por tau_sq no BayesianTau
        confidence = torch.exp(-0.5 * d_sq)

        return confidence, d_sq, tau_sq

    def confidence_with_metric(
        self,
        coords: torch.Tensor,
        truth_centroid: torch.Tensor,
        metric_distance: torch.Tensor,
    ) -> torch.Tensor:
        """Confianca usando distancia pre-computada pelo metric tensor.

        Util quando a distancia ja foi calculada pelo GeodesicDistance
        com o tensor metrico completo G.

        Args:
            coords: [B, T, drm_dim] (nao usado, mantido por interface)
            truth_centroid: [drm_dim] (nao usado)
            metric_distance: [B, T, 1] distancia geodesica

        Returns:
            confidence: [B, T, 1]
        """
        tau_sq = self.tau.get_tau_sq()
        if self.tau.per_axis:
            # Usa media dos tau_sq para distancia escalar
            avg_tau_sq = tau_sq.mean()
        else:
            avg_tau_sq = tau_sq

        d_sq = metric_distance ** 2
        confidence = torch.exp(-0.5 * d_sq / avg_tau_sq)
        return confidence
