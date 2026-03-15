"""
Dataclasses de output do modelo.

EpistemicTomography: tomografia epistemica por token (Q1, Q2, coords, phi, VI).
ModelOutput: logits + tomografia.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


@dataclass
class EpistemicTomography:
    """Tomografia epistemica por token - resultado do EpistemicHead.

    Todos os tensores tem shape [B, T, *] onde B=batch, T=seq_len.
    Campos Optional sao preenchidos apenas quando o modulo correspondente
    esta habilitado via config.
    """

    # --- Core (sempre presente) ---

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

    # --- Eidos Decay (Tier 1) ---
    eidos_weights: Optional[torch.Tensor] = None      # [B, T, 1]
    axis_balance: Optional[torch.Tensor] = None        # [B, T, 5]

    # --- Filosofia3 (Tier 1) ---
    conflict_intensity: Optional[torch.Tensor] = None  # [B, T, 1]
    mode_probs: Optional[torch.Tensor] = None          # [B, T, 4]

    # --- Consciousness (Tier 1) ---
    mood: Optional[torch.Tensor] = None                # [B, T, 1]
    curiosity: Optional[torch.Tensor] = None           # [B, T, 1]
    energy: Optional[torch.Tensor] = None              # [B, T, 1]
    drives: Optional[torch.Tensor] = None              # [B, T, 3]

    # --- Grounding (Tier 2) ---
    task_probs: Optional[torch.Tensor] = None          # [B, T, 9]
    task_confidence: Optional[torch.Tensor] = None     # [B, T, 1]
    ambiguity_level: Optional[torch.Tensor] = None     # [B, T, 1]
    ambiguity_type: Optional[torch.Tensor] = None      # [B, T, 5]

    # --- Plasticity (Tier 2) ---
    plasticity_remaining: Optional[torch.Tensor] = None  # [B, T, 1]
    gate_value: Optional[torch.Tensor] = None            # [B, T, 1]

    # --- MPL (Tier 2) ---
    frontier_score: Optional[torch.Tensor] = None      # [B, T, 1]

    # --- MOPsi (Tier 3) ---
    human_state: Optional[torch.Tensor] = None         # [B, T, 5]
    psi: Optional[torch.Tensor] = None                 # [B, T, 1]
    mediated_score: Optional[torch.Tensor] = None      # [B, T, 1]

    # --- CausalState (Tier 3) ---
    state_gate: Optional[torch.Tensor] = None          # [B, 1]

    # --- Metacognitive (Tier 3) ---
    divergence: Optional[torch.Tensor] = None          # [B, T, 1]

    # --- Metric Field (DRM) ---
    metric_G: Optional[torch.Tensor] = None            # [B, T, D, D] ou [D, D]

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict de tensores (para serializar)."""
        result = {}
        for k, v in self.__dict__.items():
            if v is not None and isinstance(v, torch.Tensor):
                result[k] = v
        return result

    def detach(self) -> "EpistemicTomography":
        """Retorna copia detached (sem grad)."""
        d = self.to_dict()
        detached = {k: v.detach() for k, v in d.items()}
        return EpistemicTomography(**detached)

    def to(self, device: torch.device) -> "EpistemicTomography":
        """Move todos os tensores para device."""
        d = self.to_dict()
        moved = {k: v.to(device) for k, v in d.items()}
        return EpistemicTomography(**moved)


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
