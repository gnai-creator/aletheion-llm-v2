"""
Dataclasses de output do modelo.

EpistemicTomography: tomografia epistemica por token (Q1, Q2, coords, phi, VI).
ModelOutput: logits + tomografia.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class EpistemicTomography:
    """Tomografia epistemica por token - resultado do EpistemicHead.

    Todos os tensores tem shape [B, T, *] onde B=batch, T=seq_len.
    """

    # Incertezas Q1/Q2
    q1: torch.Tensor               # [B, T, 1] aleatoria
    q2: torch.Tensor               # [B, T, 1] epistemica

    # MAD confidence
    confidence: torch.Tensor        # [B, T, 1] C(p) = exp(-d^2/2tau^2)

    # DRM coordenadas 5D
    drm_coords: torch.Tensor       # [B, T, 5] coordenadas no manifold

    # Directional field
    directional_dim: torch.Tensor   # [B, T, 1] dim_D soft

    # Distancia geodesica
    metric_distance: torch.Tensor   # [B, T, 1] distancia ao truth centroid

    # VI - phi components
    phi_components: torch.Tensor    # [B, T, 4] (dim, disp, ent, conf)
    phi_total: torch.Tensor         # [B, T, 1] saude global

    # VI - direcao e severidade
    vi_direction: torch.Tensor      # [B, T, 5] direcao de correcao
    vi_severity: torch.Tensor       # [B, T, 1] severidade

    # Temperatura adaptativa
    temperature: torch.Tensor       # [B, T, 1] tau adaptativo

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict de tensores (para serializar)."""
        return {
            "q1": self.q1,
            "q2": self.q2,
            "confidence": self.confidence,
            "drm_coords": self.drm_coords,
            "directional_dim": self.directional_dim,
            "metric_distance": self.metric_distance,
            "phi_components": self.phi_components,
            "phi_total": self.phi_total,
            "vi_direction": self.vi_direction,
            "vi_severity": self.vi_severity,
            "temperature": self.temperature,
        }

    def detach(self) -> "EpistemicTomography":
        """Retorna copia detached (sem grad)."""
        return EpistemicTomography(
            **{k: v.detach() for k, v in self.to_dict().items()}
        )

    def to(self, device: torch.device) -> "EpistemicTomography":
        """Move todos os tensores para device."""
        return EpistemicTomography(
            **{k: v.to(device) for k, v in self.to_dict().items()}
        )


@dataclass
class ModelOutput:
    """Output completo do AletheionV2Model."""

    logits: torch.Tensor                        # [B, T, V]
    tomography: Optional[EpistemicTomography]    # None se epistemic desabilitado
    hidden_states: torch.Tensor                  # [B, T, d_model]
    attention_patterns: Optional[torch.Tensor]   # [B, n_layers, n_heads, T, T]

    @property
    def loss_inputs(self) -> Dict[str, torch.Tensor]:
        """Retorna dict com inputs necessarios para loss composta."""
        result = {"logits": self.logits}
        if self.tomography is not None:
            result.update(self.tomography.to_dict())
        return result
