"""
PhiPsiConflictHead: Deteccao de conflito phi-psi entre tokens adjacentes.

Referencia ATIC: filosofia3/conflict_detector.py + meta_policy.py (890 linhas).
Convertido para nn.Module com ~381 params.

Computa deltas entre tokens adjacentes (phi e quality),
detecta conflito via cosine similarity analitica + MLP aprendivel,
e classifica o modo de operacao (4 modos).

Input: phi_components [B,T,4], confidence [B,T,1]
Output: conflict_intensity [B,T,1], mode_probs [B,T,4]

4 Modos:
  0 - ALIGNED: phi e psi alinham, operacao normal
  1 - CONFLICT_TOLERATED: conflito leve, toleravel
  2 - SIGNAL_HUMAN: conflito significativo, sinaliza
  3 - RECOVERY: conflito severo, modo recuperacao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConflictOutput:
    """Output do PhiPsiConflictHead."""

    conflict_intensity: torch.Tensor   # [B, T, 1] intensidade do conflito [0,1]
    mode_probs: torch.Tensor           # [B, T, 4] probabilidades dos 4 modos
    conflict_analytical: torch.Tensor  # [B, T, 1] componente analitica (diagnostico)


class PhiPsiConflictHead(nn.Module):
    """Deteccao de conflito phi-psi via blend analitico/aprendivel.

    Computa deltas entre tokens adjacentes nas componentes phi (4D)
    e quality (confianca projetada para 4D). O conflito e medido
    como combinacao de cosine similarity analitica e MLP aprendivel.

    Args:
        quality_projection: pesos de projecao de confianca para 4D
        analytical_weight: peso da componente analitica no blend
    """

    # Modos de operacao
    ALIGNED = 0
    CONFLICT_TOLERATED = 1
    SIGNAL_HUMAN = 2
    RECOVERY = 3

    def __init__(
        self,
        quality_projection: tuple = (0.1, 0.1, 0.1, 0.7),
        analytical_weight: float = 0.7,
    ):
        super().__init__()
        self.analytical_weight = analytical_weight
        self.learned_weight = 1.0 - analytical_weight

        # Buffer para projecao de quality
        self.register_buffer(
            "quality_proj",
            torch.tensor(quality_projection, dtype=torch.float32),
        )

        # MLP para componente aprendivel do conflito
        # Input: concat(delta_phi[4], delta_quality[4]) = 8D
        self.conflict_proj = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # MLP para classificacao de modo
        # Input: mesmos 8D deltas
        self.mode_head = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(
        self,
        phi_components: torch.Tensor,
        confidence: torch.Tensor,
    ) -> ConflictOutput:
        """Computa conflito e modo de operacao.

        Args:
            phi_components: [B, T, 4] (dim, disp, ent, conf)
            confidence: [B, T, 1] confianca MAD

        Returns:
            ConflictOutput com intensity, modes e diagnostico
        """
        B, T, _ = phi_components.shape

        # Computa deltas entre tokens adjacentes
        # Pad token 0 com zeros para manter T
        if T > 1:
            delta_phi = phi_components[:, 1:, :] - phi_components[:, :-1, :]
            # Pad no inicio: [B, 1, 4] zeros
            pad = torch.zeros(B, 1, 4, device=phi_components.device)
            delta_phi = torch.cat([pad, delta_phi], dim=1)  # [B, T, 4]
        else:
            delta_phi = torch.zeros_like(phi_components)

        # Projeta confianca para 4D
        # [B, T, 1] * [4] -> [B, T, 4]
        quality_4d = confidence * self.quality_proj.unsqueeze(0).unsqueeze(0)

        if T > 1:
            delta_quality = quality_4d[:, 1:, :] - quality_4d[:, :-1, :]
            pad_q = torch.zeros(B, 1, 4, device=phi_components.device)
            delta_quality = torch.cat([pad_q, delta_quality], dim=1)
        else:
            delta_quality = torch.zeros_like(quality_4d)

        # Cosine similarity analitica entre delta_phi e delta_quality
        # [B, T, 1]
        cos_sim = F.cosine_similarity(delta_phi, delta_quality, dim=-1, eps=1e-8)
        # Converte de [-1, 1] para [0, 1]: conflito = (1 - cos) / 2
        conflict_analytical = ((1.0 - cos_sim) / 2.0).unsqueeze(-1)  # [B, T, 1]

        # Componente aprendivel
        # Concat deltas: [B, T, 8]
        delta_concat = torch.cat([delta_phi, delta_quality], dim=-1)
        conflict_learned = self.conflict_proj(delta_concat)  # [B, T, 1]

        # Blend: analitico + aprendivel
        conflict_intensity = (
            self.analytical_weight * conflict_analytical
            + self.learned_weight * conflict_learned
        )

        # Classificacao de modo: [B, T, 4]
        mode_logits = self.mode_head(delta_concat)
        mode_probs = F.softmax(mode_logits, dim=-1)

        return ConflictOutput(
            conflict_intensity=conflict_intensity,
            mode_probs=mode_probs,
            conflict_analytical=conflict_analytical,
        )

    def extra_repr(self) -> str:
        return (
            f"analytical_weight={self.analytical_weight}, "
            f"modes=4 (ALIGNED/TOLERATED/SIGNAL/RECOVERY)"
        )
