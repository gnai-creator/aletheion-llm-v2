"""
Geodesic Distance: distancia no manifold epistemico.

Dois modos:
- G constante [D,D]: Mahalanobis direta (espaco plano)
- G(x) via MetricNet: integral de linha com quadratura (curvatura real)

A integral de linha ao longo do segmento reto nao e a geodesica
verdadeira (que minimiza comprimento), mas captura a variacao
de G(x) ao longo do caminho e e diferenciavel.
"""

import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aletheion_v2.drm.metric_tensor import MetricNet


class GeodesicDistance(nn.Module):
    """Calcula distancia no manifold epistemico.

    Suporta G constante (Mahalanobis) e G(x) variavel
    (integral de linha via MetricNet).

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
        metric_net: Optional["MetricNet"] = None,
    ) -> torch.Tensor:
        """Computa distancia ao truth centroid.

        Se metric_net e fornecido, usa integral de linha (curvatura real).
        Caso contrario, usa Mahalanobis com G constante.

        Args:
            coords: [B, T, drm_dim] coordenadas dos tokens
            truth_centroid: [drm_dim] ponto de referencia truth
            G: [D, D] tensor metrico constante (fallback)
            metric_net: MetricNet para G(x) variavel (opcional)

        Returns:
            distance: [B, T, 1] distancia no manifold
        """
        if metric_net is not None:
            return metric_net.line_integral_distance(coords, truth_centroid)

        # Fallback: Mahalanobis com G constante
        delta = coords - truth_centroid.unsqueeze(0).unsqueeze(0)
        Gd = torch.matmul(delta, G)
        d_sq = (Gd * delta).sum(dim=-1, keepdim=True)
        d_sq = d_sq.clamp(min=0.0)
        return torch.sqrt(d_sq + self.eps)

    def pairwise(
        self,
        coords_a: torch.Tensor,
        coords_b: torch.Tensor,
        G: torch.Tensor,
        metric_net: Optional["MetricNet"] = None,
    ) -> torch.Tensor:
        """Distancia entre dois conjuntos de pontos.

        Args:
            coords_a: [B, T, D]
            coords_b: [B, T, D]
            G: [D, D] (fallback)
            metric_net: MetricNet para G(x) (opcional)

        Returns:
            distance: [B, T, 1]
        """
        if metric_net is not None:
            return metric_net.line_integral_distance(coords_a, coords_b)

        delta = coords_a - coords_b
        Gd = torch.matmul(delta, G)
        d_sq = (Gd * delta).sum(dim=-1, keepdim=True).clamp(min=0.0)
        return torch.sqrt(d_sq + self.eps)

    def batch_to_anchors(
        self,
        coords: torch.Tensor,
        anchors: torch.Tensor,
        G: torch.Tensor,
        metric_net: Optional["MetricNet"] = None,
    ) -> torch.Tensor:
        """Distancia de cada token a cada anchor.

        Para metric_net, usa G avaliado nas coords do token
        (nao integral de linha completa -- custo O(A) em vez de O(A*n_quad)).

        Args:
            coords: [B, T, D]
            anchors: [A, D]
            G: [D, D] (fallback)
            metric_net: MetricNet para G(x) (opcional)

        Returns:
            distances: [B, T, A]
        """
        if metric_net is not None:
            # Usa G local no ponto do token como aproximacao
            G_local = metric_net(coords)  # [B, T, D, D]
            anchors_exp = anchors.unsqueeze(0).unsqueeze(0)  # [1, 1, A, D]
            coords_exp = coords.unsqueeze(2)  # [B, T, 1, D]
            delta = coords_exp - anchors_exp  # [B, T, A, D]

            # G_local e [B,T,D,D], precisa expandir para [B,T,1,D,D]
            G_exp = G_local.unsqueeze(2)  # [B, T, 1, D, D]
            Gd = torch.matmul(delta.unsqueeze(-2), G_exp).squeeze(-2)
            d_sq = (Gd * delta).sum(dim=-1).clamp(min=0.0)
            return torch.sqrt(d_sq + self.eps)

        # Fallback: Mahalanobis constante
        anchors_exp = anchors.unsqueeze(0).unsqueeze(0)
        coords_exp = coords.unsqueeze(2)
        delta = coords_exp - anchors_exp
        Gd = torch.matmul(delta, G)
        d_sq = (Gd * delta).sum(dim=-1).clamp(min=0.0)
        return torch.sqrt(d_sq + self.eps)
