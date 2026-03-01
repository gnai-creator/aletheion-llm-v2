"""
Intentionality Vector (VI): direcao e severidade de correcao.

Quando phi(M) cai abaixo do limiar critico, o VI computa:
    - direction: vetor de correcao no manifold 5D
    - severity: intensidade da degradacao (sqrt para resposta agressiva)

O VI age como homeostase do manifold epistemico.

Formulas:
    severity = sqrt(max(0, 1 - phi_total / phi_critical))
    direction = proj(phi_components, coords)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IntentionalityVector(nn.Module):
    """Vetor de Intencionalidade - correcao homestatica do manifold.

    Quando phi(M) < phi_critical, computa direcao e severidade
    de correcao para restaurar saude do manifold.

    Args:
        drm_dim: Dimensao do manifold (5)
        phi_critical: Limiar critico de phi
        injection_strength: Forca da injecao corretiva
        confidence_penalty: Penalidade maxima na confianca
    """

    def __init__(
        self,
        drm_dim: int = 5,
        phi_critical: float = 0.5,
        injection_strength: float = 0.6,
        confidence_penalty: float = 0.4,
    ):
        super().__init__()
        self.drm_dim = drm_dim
        self.phi_critical = phi_critical
        self.injection_strength = injection_strength
        self.confidence_penalty = confidence_penalty

        # Rede para computar direcao de correcao
        # Input: phi_components (4) + coords (5) = 9
        self.direction_net = nn.Sequential(
            nn.Linear(4 + drm_dim, drm_dim * 2),
            nn.GELU(),
            nn.Linear(drm_dim * 2, drm_dim),
        )

        # Rede para modular severidade
        # Input: phi_components (4)
        self.severity_net = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def _compute_severity_analytical(
        self, phi_total: torch.Tensor
    ) -> torch.Tensor:
        """Severidade analitica (sqrt curve).

        severity = sqrt(max(0, 1 - phi/phi_critical))

        Args:
            phi_total: [B, T, 1]

        Returns:
            severity: [B, T, 1] em [0, 1]
        """
        ratio = phi_total / self.phi_critical
        raw = (1.0 - ratio).clamp(min=0.0)
        return torch.sqrt(raw)

    def forward(
        self,
        phi_components: torch.Tensor,
        phi_total: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computa direcao e severidade do VI.

        Args:
            phi_components: [B, T, 4] componentes de phi
            phi_total: [B, T, 1] phi total
            coords: [B, T, drm_dim] coordenadas atuais

        Returns:
            vi_direction: [B, T, drm_dim] direcao de correcao
            vi_severity: [B, T, 1] severidade
        """
        # Severidade: componente analitica + componente aprendida
        severity_analytical = self._compute_severity_analytical(phi_total)
        severity_learned = self.severity_net(phi_components)
        # Blend: 70% analitico, 30% aprendido
        severity = 0.7 * severity_analytical + 0.3 * severity_learned

        # Direcao de correcao
        dir_input = torch.cat([phi_components, coords], dim=-1)  # [B, T, 9]
        direction_raw = self.direction_net(dir_input)  # [B, T, drm_dim]

        # Normaliza direcao e escala por severity
        direction_norm = F.normalize(direction_raw, dim=-1)
        vi_direction = direction_norm * severity * self.injection_strength

        return vi_direction, severity

    def correct_confidence(
        self,
        confidence: torch.Tensor,
        severity: torch.Tensor,
    ) -> torch.Tensor:
        """Aplica penalidade na confianca baseada na severidade.

        conf' = conf * (1 - severity * penalty_max)

        Args:
            confidence: [B, T, 1] confianca original
            severity: [B, T, 1] severidade do VI

        Returns:
            corrected: [B, T, 1] confianca corrigida
        """
        penalty = severity * self.confidence_penalty
        corrected = confidence * (1.0 - penalty)
        return corrected.clamp(min=0.0, max=1.0)
