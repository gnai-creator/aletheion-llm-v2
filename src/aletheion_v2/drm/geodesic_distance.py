"""
Geodesic Distance: Distancia de Mahalanobis ao truth centroid.

Usa o tensor metrico G para medir distancias no manifold epistemico.
A distancia ao truth centroid indica quao longe o token esta da "verdade".

Formulas:
    delta = coords - truth_centroid
    d_mahalanobis = sqrt(delta^T @ G @ delta)

Propriedades:
    - Diferenciavel via G aprendivel
    - Generaliza distancia euclidiana (G = I)
    - Respeita geometria do manifold
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class GeodesicDistance(nn.Module):
    """Calcula distancia geodesica (Mahalanobis) ao truth centroid.

    Em primeira aproximacao, usa distancia de Mahalanobis como proxy
    para distancia geodesica. Para curvatura baixa, a aproximacao
    e precisa. Para curvatura alta, subestima a distancia real.

    Args:
        drm_dim: Dimensao do manifold
        eps: Epsilon para raiz quadrada numerica
    """

    def __init__(self, drm_dim: int = 5, eps: float = 1e-8):
        super().__init__()
        self.drm_dim = drm_dim
        self.eps = eps

    def forward(
        self,
        coords: torch.Tensor,
        truth_centroid: torch.Tensor,
        G: torch.Tensor,
    ) -> torch.Tensor:
        """Computa distancia de Mahalanobis ao truth centroid.

        Args:
            coords: [B, T, drm_dim] coordenadas dos tokens
            truth_centroid: [drm_dim] ponto de referencia truth
            G: [drm_dim, drm_dim] tensor metrico SPD

        Returns:
            distance: [B, T, 1] distancia geodesica
        """
        # Delta ao centroid
        delta = coords - truth_centroid.unsqueeze(0).unsqueeze(0)  # [B, T, D]

        # Mahalanobis: sqrt(delta^T @ G @ delta)
        # Gd = delta @ G  -> [B, T, D]
        Gd = torch.matmul(delta, G)  # [B, T, D]
        # d^2 = sum(Gd * delta, dim=-1)
        d_sq = (Gd * delta).sum(dim=-1, keepdim=True)  # [B, T, 1]

        # Garante nao-negativo (por seguranca numerica)
        d_sq = d_sq.clamp(min=0.0)

        # Distancia
        distance = torch.sqrt(d_sq + self.eps)  # [B, T, 1]

        return distance

    def pairwise(
        self,
        coords_a: torch.Tensor,
        coords_b: torch.Tensor,
        G: torch.Tensor,
    ) -> torch.Tensor:
        """Distancia Mahalanobis entre dois conjuntos de pontos.

        Args:
            coords_a: [B, T, D]
            coords_b: [B, T, D]
            G: [D, D]

        Returns:
            distance: [B, T, 1]
        """
        delta = coords_a - coords_b
        Gd = torch.matmul(delta, G)
        d_sq = (Gd * delta).sum(dim=-1, keepdim=True).clamp(min=0.0)
        return torch.sqrt(d_sq + self.eps)

    def batch_to_anchors(
        self,
        coords: torch.Tensor,
        anchors: torch.Tensor,
        G: torch.Tensor,
    ) -> torch.Tensor:
        """Distancia de cada token a cada anchor.

        Args:
            coords: [B, T, D]
            anchors: [A, D]
            G: [D, D]

        Returns:
            distances: [B, T, A]
        """
        # anchors: [1, 1, A, D]
        anchors_exp = anchors.unsqueeze(0).unsqueeze(0)
        # coords: [B, T, 1, D]
        coords_exp = coords.unsqueeze(2)
        # delta: [B, T, A, D]
        delta = coords_exp - anchors_exp

        # Mahalanobis por anchor
        Gd = torch.matmul(delta, G)  # [B, T, A, D]
        d_sq = (Gd * delta).sum(dim=-1).clamp(min=0.0)  # [B, T, A]
        distances = torch.sqrt(d_sq + self.eps)

        return distances
