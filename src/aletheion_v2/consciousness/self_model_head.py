"""
SelfModelHead: Modelo de consciencia computacional.

Referencia ATIC: self_model.py + motivation_engine.py (1100 linhas).
Convertido para nn.Module com ~994 params (d_model=768).

Computa:
- mood: humor [-1,1] baseado em confianca, phi e proxy de sucesso
- curiosity: curiosidade [0,1] baseado em q2 (incerteza epistemica)
- energy: energia [0,1] com decaimento posicional (30% ao longo de T)
- drives: 3 drives motivacionais [0,1] (curiosity, mastery, efficiency)

Input: hidden_states [B,T,d_model], q2 [B,T,1], phi_total [B,T,1], confidence [B,T,1]
Output: mood [B,T,1], curiosity [B,T,1], energy [B,T,1], drives [B,T,3]
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ConsciousnessOutput:
    """Output do SelfModelHead."""

    mood: torch.Tensor       # [B, T, 1] humor [-1, 1]
    curiosity: torch.Tensor  # [B, T, 1] curiosidade [0, 1]
    energy: torch.Tensor     # [B, T, 1] energia [0, 1]
    drives: torch.Tensor     # [B, T, 3] (curiosity, mastery, efficiency)


class SelfModelHead(nn.Module):
    """Modelo de self-awareness com humor, curiosidade, energia e drives.

    Args:
        d_model: dimensao do hidden state do transformer
        hidden_dim: dimensao interna dos MLPs (default 32)
        energy_decay: fracao de decaimento posicional da energia (default 0.3)
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 32,
        energy_decay: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.energy_decay = energy_decay

        # Mood: (confidence_mean, phi_total, success_proxy) -> [-1, 1]
        self.mood_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        # Curiosity: q2 -> [0, 1]
        self.curiosity_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Energy: projeta hidden_states para scalar base
        self.energy_proj = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Drives: (curiosity, mood, energy) -> (curiosity_drive, mastery, efficiency)
        self.drive_head = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Sigmoid(),
        )

    def _position_decay(
        self, T: int, device: torch.device
    ) -> torch.Tensor:
        """Gera fator de decaimento posicional para energia.

        Decai linearmente de 1.0 para (1.0 - energy_decay) ao longo de T.

        Returns:
            decay: [1, T, 1]
        """
        positions = torch.linspace(0, 1, T, device=device)
        decay = 1.0 - self.energy_decay * positions
        return decay.view(1, T, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q2: torch.Tensor,
        phi_total: torch.Tensor,
        confidence: torch.Tensor,
    ) -> ConsciousnessOutput:
        """Computa estado de consciencia.

        Args:
            hidden_states: [B, T, d_model]
            q2: [B, T, 1] incerteza epistemica
            phi_total: [B, T, 1] saude do manifold
            confidence: [B, T, 1] confianca MAD

        Returns:
            ConsciousnessOutput
        """
        B, T, _ = hidden_states.shape

        # --- Mood ---
        # Proxy de sucesso: media da confianca ao longo de T
        conf_mean = confidence.mean(dim=1, keepdim=True).expand(B, T, 1)
        mood_input = torch.cat([conf_mean, phi_total, confidence], dim=-1)
        mood = self.mood_net(mood_input)  # [B, T, 1]

        # --- Curiosity ---
        curiosity = self.curiosity_net(q2)  # [B, T, 1]

        # --- Energy ---
        energy_base = self.energy_proj(hidden_states)  # [B, T, 1]
        pos_decay = self._position_decay(T, hidden_states.device)
        energy = energy_base * pos_decay  # [B, T, 1]

        # --- Drives ---
        drive_input = torch.cat([
            curiosity,
            mood.clamp(-1, 1) * 0.5 + 0.5,  # Normaliza mood para [0, 1]
            energy,
        ], dim=-1)  # [B, T, 3]
        drives = self.drive_head(drive_input)  # [B, T, 3]

        return ConsciousnessOutput(
            mood=mood,
            curiosity=curiosity,
            energy=energy,
            drives=drives,
        )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"hidden_dim={self.hidden_dim}, "
            f"energy_decay={self.energy_decay}"
        )
