"""
Manifold Embedding: Projeta hidden_states em coordenadas 5D.

Mapeia representacoes do transformer para o espaco epistemico 5D:
    [q1_aleatoric, q2_epistemic, domain_complexity, temporal_relevance, response_quality]

Anchors sao pontos fixos que definem o sistema de coordenadas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# Nomes dos 5 eixos epistemicos
AXIS_NAMES = [
    "q1_aleatoric",       # Incerteza aleatoria
    "q2_epistemic",       # Incerteza epistemica
    "domain_complexity",  # Complexidade do dominio
    "temporal_relevance", # Relevancia temporal
    "response_quality",   # Qualidade da resposta
]


class AnchorPoints(nn.Module):
    """Pontos ancora fixos no manifold epistemico.

    Define 6 pontos de referencia que formam o sistema de coordenadas:
    - truth: alta qualidade, baixa incerteza
    - ignorance: alta incerteza epistemica
    - noise: alta incerteza aleatoria
    - complex: alta complexidade de dominio
    - stale: baixa relevancia temporal
    - ideal: ponto de referencia ideal (centro)
    """

    def __init__(self, drm_dim: int = 5, num_anchors: int = 6):
        super().__init__()
        self.drm_dim = drm_dim
        self.num_anchors = num_anchors

        # Anchors fixos (nao treinaveis)
        anchors = torch.zeros(num_anchors, drm_dim)

        # truth: baixa incerteza, alta qualidade
        anchors[0] = torch.tensor([0.1, 0.1, 0.5, 0.9, 0.9])
        # ignorance: alta incerteza epistemica
        anchors[1] = torch.tensor([0.3, 0.9, 0.5, 0.5, 0.2])
        # noise: alta incerteza aleatoria
        anchors[2] = torch.tensor([0.9, 0.3, 0.5, 0.5, 0.3])
        # complex: alta complexidade
        anchors[3] = torch.tensor([0.5, 0.5, 0.9, 0.5, 0.5])
        # stale: baixa relevancia temporal
        anchors[4] = torch.tensor([0.3, 0.3, 0.5, 0.1, 0.4])
        # ideal: referencia equilibrada
        anchors[5] = torch.tensor([0.2, 0.2, 0.3, 0.8, 0.8])

        self.register_buffer("anchors", anchors)

    @property
    def truth_centroid(self) -> torch.Tensor:
        """Retorna o anchor truth (indice 0)."""
        return self.anchors[0]


class ManifoldEmbedding(nn.Module):
    """Projeta hidden_states para coordenadas 5D no manifold epistemico.

    Arquitetura:
        hidden [B, T, d_model]
            |
            v
        projection (Linear d_model -> drm_dim * 4)
            |
            v
        GELU + Linear -> raw_coords [B, T, drm_dim]
            |
            v
        Sigmoid -> coords [B, T, drm_dim] em [0, 1]
            |
            v
        anchor_distances [B, T, num_anchors]

    As coordenadas sao enriquecidas com distancias relativas aos anchors
    para fornecer contexto geometrico ao restante do pipeline.
    """

    def __init__(
        self,
        d_model: int,
        drm_dim: int = 5,
        num_anchors: int = 6,
        hidden_factor: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.drm_dim = drm_dim
        hidden_dim = drm_dim * hidden_factor

        # Projecao nao-linear para coords 5D
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, drm_dim),
            nn.Sigmoid(),  # Coords em [0, 1]
        )

        # Anchors fixos
        self.anchors = AnchorPoints(drm_dim, num_anchors)

        # Projecao de distancia-anchor para feature
        self.anchor_proj = nn.Linear(num_anchors, drm_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializacao dos pesos."""
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.anchor_proj.weight)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Projeta hidden_states para coordenadas 5D.

        Args:
            hidden_states: [B, T, d_model]

        Returns:
            coords: [B, T, drm_dim] coordenadas no manifold
            anchor_dists: [B, T, num_anchors] distancias aos anchors
        """
        # Projecao para coords 5D
        coords = self.proj(hidden_states)  # [B, T, drm_dim]

        # Distancias euclidianas aos anchors
        # coords: [B, T, drm_dim], anchors: [num_anchors, drm_dim]
        anchors = self.anchors.anchors.unsqueeze(0).unsqueeze(0)  # [1, 1, A, D]
        coords_exp = coords.unsqueeze(2)  # [B, T, 1, D]
        diffs = coords_exp - anchors  # [B, T, A, D]
        anchor_dists = torch.norm(diffs, dim=-1)  # [B, T, A]

        # Refinamento residual via anchor distances
        anchor_features = self.anchor_proj(anchor_dists)  # [B, T, drm_dim]
        coords = torch.sigmoid(
            torch.logit(coords.clamp(1e-6, 1 - 1e-6)) + 0.1 * anchor_features
        )

        return coords, anchor_dists
