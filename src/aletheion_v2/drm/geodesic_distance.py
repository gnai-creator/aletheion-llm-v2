"""
Geodesic Distance: distancia no manifold epistemico.

Dois modos:
- G constante [D,D]: Mahalanobis direta (espaco plano)
- G(x) via MetricNet: integral de linha com quadratura (curvatura real)

Gamma-scaling opcional (Proposicao 4.2 DRM Relativistic Dynamics):
  gamma(v) = 1/sqrt(1 - v^2/c^2)
  Escala distancias por gamma baseado na distancia ao anchor mais proximo.
  Regioes longe dos anchors (alta incerteza) recebem maior resolucao metrica.

A integral de linha ao longo do segmento reto nao e a geodesica
verdadeira (que minimiza comprimento), mas captura a variacao
de G(x) ao longo do caminho e e diferenciavel.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aletheion_v2.drm.metric_tensor import MetricNet


def _gamma_scale(
    coords: torch.Tensor,
    anchors: torch.Tensor,
    c_param: float = math.sqrt(5),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Fator de escala gamma baseado na distancia ao anchor mais proximo.

    Mapeia distancia euclidiana ao anchor mais proximo para v in [0, c),
    e retorna gamma(v) = 1/sqrt(1 - v^2/c^2).

    c_param default = sqrt(5) ~= 2.236 (diagonal maxima em [0,1]^5).

    Args:
        coords: [B, T, D] coordenadas dos tokens.
        anchors: [A, D] coordenadas dos anchors.
        c_param: Velocidade limite (distancia maxima no manifold).
        eps: Epsilon numerico.

    Returns:
        gamma: [B, T, 1] fator de escala >= 1.0.
    """
    # coords: [B, T, 1, D], anchors: [1, 1, A, D]
    delta = coords.unsqueeze(2) - anchors.unsqueeze(0).unsqueeze(0)
    dists = delta.norm(dim=-1)  # [B, T, A]
    v = dists.min(dim=-1, keepdim=True).values  # [B, T, 1]

    # Clamp para v < c (nunca atingir c)
    v = v.clamp(max=c_param * 0.999)

    # gamma = 1 / sqrt(1 - v^2/c^2)
    gamma = 1.0 / torch.sqrt(1.0 - (v / c_param) ** 2 + eps)

    return gamma


class GeodesicDistance(nn.Module):
    """Calcula distancia no manifold epistemico.

    Suporta G constante (Mahalanobis) e G(x) variavel
    (integral de linha via MetricNet).

    Gamma-scaling opcional escala distancias por fator relativistic
    baseado na distancia ao anchor mais proximo.

    Args:
        drm_dim: Dimensao do manifold
        eps: Epsilon para raiz quadrada numerica
        gamma_enabled: Ativar gamma-scaling relativistic
        gamma_c_param: Velocidade limite c (default sqrt(5))
    """

    def __init__(
        self,
        drm_dim: int = 5,
        eps: float = 1e-8,
        gamma_enabled: bool = False,
        gamma_c_param: float = math.sqrt(5),
    ):
        super().__init__()
        self.drm_dim = drm_dim
        self.eps = eps
        self.gamma_enabled = gamma_enabled
        self.gamma_c_param = gamma_c_param

    def forward(
        self,
        coords: torch.Tensor,
        truth_centroid: torch.Tensor,
        G: torch.Tensor,
        metric_net: Optional["MetricNet"] = None,
        anchors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computa distancia ao truth centroid.

        Se metric_net e fornecido, usa integral de linha (curvatura real).
        Caso contrario, usa Mahalanobis com G constante.

        Args:
            coords: [B, T, drm_dim] coordenadas dos tokens
            truth_centroid: [drm_dim] ponto de referencia truth
            G: [D, D] tensor metrico constante (fallback)
            metric_net: MetricNet para G(x) variavel (opcional)
            anchors: [A, D] anchors para gamma-scaling (opcional)

        Returns:
            distance: [B, T, 1] distancia no manifold
        """
        if metric_net is not None:
            dist = metric_net.line_integral_distance(coords, truth_centroid)
        else:
            # Fallback: Mahalanobis com G constante
            delta = coords - truth_centroid.unsqueeze(0).unsqueeze(0)
            Gd = torch.matmul(delta, G)
            d_sq = (Gd * delta).sum(dim=-1, keepdim=True)
            d_sq = d_sq.clamp(min=0.0)
            dist = torch.sqrt(d_sq + self.eps)

        # Gamma-scaling: escala distancia por fator relativistic
        if self.gamma_enabled and anchors is not None:
            gamma = _gamma_scale(
                coords, anchors, c_param=self.gamma_c_param,
            )
            dist = dist * gamma

        return dist

    def pairwise(
        self,
        coords_a: torch.Tensor,
        coords_b: torch.Tensor,
        G: torch.Tensor,
        metric_net: Optional["MetricNet"] = None,
        anchors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Distancia entre dois conjuntos de pontos.

        Args:
            coords_a: [B, T, D]
            coords_b: [B, T, D]
            G: [D, D] (fallback)
            metric_net: MetricNet para G(x) (opcional)
            anchors: [A, D] anchors para gamma-scaling (opcional)

        Returns:
            distance: [B, T, 1]
        """
        if metric_net is not None:
            dist = metric_net.line_integral_distance(coords_a, coords_b)
        else:
            delta = coords_a - coords_b
            Gd = torch.matmul(delta, G)
            d_sq = (Gd * delta).sum(dim=-1, keepdim=True).clamp(min=0.0)
            dist = torch.sqrt(d_sq + self.eps)

        if self.gamma_enabled and anchors is not None:
            # Usa midpoint dos dois pontos para computar gamma
            midpoint = (coords_a + coords_b) * 0.5
            gamma = _gamma_scale(
                midpoint, anchors, c_param=self.gamma_c_param,
            )
            dist = dist * gamma

        return dist

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

        Gamma-scaling aplicado por token (distancia ao anchor mais proximo
        escala todas as distancias desse token).

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
            dist = torch.sqrt(d_sq + self.eps)
        else:
            # Fallback: Mahalanobis constante
            anchors_exp = anchors.unsqueeze(0).unsqueeze(0)
            coords_exp = coords.unsqueeze(2)
            delta = coords_exp - anchors_exp
            Gd = torch.matmul(delta, G)
            d_sq = (Gd * delta).sum(dim=-1).clamp(min=0.0)
            dist = torch.sqrt(d_sq + self.eps)

        if self.gamma_enabled:
            # gamma [B, T, 1] -> broadcast para [B, T, A]
            gamma = _gamma_scale(
                coords, anchors, c_param=self.gamma_c_param,
            )
            dist = dist * gamma  # [B, T, A] * [B, T, 1]

        return dist
