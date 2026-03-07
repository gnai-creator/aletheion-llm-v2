"""
FrontierHead: scoring de fronteira para exploracao do manifold.

Referencia ATIC: drm/mpl_frontier.py.
nn.Module com ~129 params.

Combina coordenadas 5D com densidade para produzir frontier_score.
Regioes de baixa densidade com coordenadas novas recebem score alto.

Input: coords [B,T,5], density [B,T,1]
Output: frontier_score [B,T,1], novelty_enhanced [B,T,1]
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class FrontierOutput:
    """Output do FrontierHead."""

    frontier_score: torch.Tensor      # [B, T, 1] score de fronteira [0, 1]
    novelty_enhanced: torch.Tensor    # [B, T, 1] novidade aprimorada


class FrontierHead(nn.Module):
    """Scoring de fronteira para exploracao do manifold.

    Aprende a identificar regioes de fronteira (alta novidade,
    baixa densidade) no espaco epistemico 5D.

    Args:
        drm_dim: dimensoes do manifold (default 5)
        hidden_dim: dimensao interna (default 16)
    """

    def __init__(self, drm_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.drm_dim = drm_dim

        # MLP: (coords[5] + density[1]) -> frontier_score
        self.scorer = nn.Sequential(
            nn.Linear(drm_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        coords: torch.Tensor,
        density: torch.Tensor,
    ) -> FrontierOutput:
        """Computa frontier score.

        Args:
            coords: [B, T, 5] coordenadas no manifold
            density: [B, T, 1] densidade local

        Returns:
            FrontierOutput com scores
        """
        # Novidade = 1 - density
        novelty = 1.0 - density  # [B, T, 1]

        # Concat: [B, T, 6]
        features = torch.cat([coords, density], dim=-1)

        # Score aprendivel
        raw_score = self.scorer(features)  # [B, T, 1]

        # Frontier score final: modula por novidade
        frontier_score = raw_score * novelty

        # Novidade aprimorada: combina raw + novidade pura
        novelty_enhanced = 0.7 * frontier_score + 0.3 * novelty

        return FrontierOutput(
            frontier_score=frontier_score,
            novelty_enhanced=novelty_enhanced,
        )

    def extra_repr(self) -> str:
        return f"drm_dim={self.drm_dim}"
