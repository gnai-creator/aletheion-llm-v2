"""
EidosDecay: Modulo de balanceamento de eixos do manifold 5D.

Referencia ATIC: drm/eidos_decay.py (deterministico, 555 linhas).
Aqui convertido para nn.Module diferenciavel com ~365 params.

Logica invertida do EidosDB original:
- Eixos sobre-representados (std alta) recebem decay mais forte
- Eixos sub-representados (std baixa) recebem reforco
- Dream mode amplifica ambos durante treinamento

Input: coords [B,T,5], confidence [B,T,1]
Output: eidos_weights [B,T,1], axis_balance [B,T,5]
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class EidosOutput:
    """Output do EidosDecay."""

    eidos_weights: torch.Tensor    # [B, T, 1] peso scalar por token
    axis_balance: torch.Tensor     # [B, T, 5] balanceamento por eixo
    axis_std: torch.Tensor         # [B, 5] desvio padrao por eixo (diagnostico)


class EidosDecay(nn.Module):
    """Balanceamento de eixos do manifold via decay/reinforce aprendiveis.

    Eixos sobre-representados (alta variancia) recebem decay,
    eixos sub-representados (baixa variancia) recebem reforco.
    O resultado e um peso scalar que modula a confianca.

    Args:
        drm_dim: dimensoes do manifold (default 5)
        base_decay: fator de decay base (default 0.95)
        base_reinforce: fator de reforco base (default 1.05)
        dream_intensity: multiplicador em dream mode (default 3.0)
    """

    def __init__(
        self,
        drm_dim: int = 5,
        base_decay: float = 0.95,
        base_reinforce: float = 1.05,
        dream_intensity: float = 3.0,
    ):
        super().__init__()
        self.drm_dim = drm_dim
        self.base_decay = base_decay
        self.base_reinforce = base_reinforce
        self.dream_intensity = dream_intensity

        # MLP para decay de eixos sobre-representados
        # Input: std por eixo [5] -> pesos de decay [5]
        self.decay_net = nn.Sequential(
            nn.Linear(drm_dim, 16),
            nn.ReLU(),
            nn.Linear(16, drm_dim),
            nn.Sigmoid(),
        )

        # MLP para reforco de eixos sub-representados
        self.reinforce_net = nn.Sequential(
            nn.Linear(drm_dim, 16),
            nn.ReLU(),
            nn.Linear(16, drm_dim),
            nn.Sigmoid(),
        )

        # Projecao para peso scalar final
        self.weight_proj = nn.Sequential(
            nn.Linear(drm_dim, 1),
            nn.Sigmoid(),
        )

        # Threshold para classificar sobre/sub-representado
        self._balance_target = 1.0 / drm_dim  # Distribuicao uniforme ideal

    def forward(
        self,
        coords: torch.Tensor,
        confidence: torch.Tensor,
        dream_mode: bool = False,
    ) -> EidosOutput:
        """Computa balanceamento de eixos.

        Args:
            coords: [B, T, 5] coordenadas no manifold
            confidence: [B, T, 1] confianca MAD
            dream_mode: se True, amplifica por dream_intensity

        Returns:
            EidosOutput com weights, balance e diagnostico
        """
        B, T, D = coords.shape

        # Computa distribuicao de eixos via std across T
        # [B, 5] - desvio padrao temporal de cada eixo
        axis_std = coords.std(dim=1)  # [B, D]

        # Normaliza std para [0, 1] (relativo ao max)
        std_max = axis_std.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        std_norm = axis_std / std_max  # [B, D]

        # Classifica: acima da media = sobre-representado
        std_mean = std_norm.mean(dim=-1, keepdim=True)  # [B, 1]
        over_mask = (std_norm > std_mean).float()  # [B, D]
        under_mask = 1.0 - over_mask

        # Computa fatores de decay e reforco
        decay_factors = self.decay_net(std_norm)    # [B, D] em [0, 1]
        reinforce_factors = self.reinforce_net(std_norm)  # [B, D] em [0, 1]

        # Escala para range correto
        # decay: [base_decay, 1.0], reinforce: [1.0, base_reinforce]
        decay_scaled = self.base_decay + decay_factors * (1.0 - self.base_decay)
        reinforce_scaled = 1.0 + reinforce_factors * (self.base_reinforce - 1.0)

        # Aplica mascara: decay para sobre, reinforce para sub
        axis_balance = over_mask * decay_scaled + under_mask * reinforce_scaled
        # [B, D]

        # Dream mode: amplifica desvio do 1.0
        if dream_mode:
            deviation = axis_balance - 1.0
            axis_balance = 1.0 + deviation * self.dream_intensity

        # Expande para [B, T, D]
        axis_balance = axis_balance.unsqueeze(1).expand(B, T, D)

        # Peso scalar por token
        # Usa a media do balance como feature
        balance_mean = axis_balance.mean(dim=-1, keepdim=True)  # [B, T, 1]
        balance_features = axis_balance[:, :, :]  # [B, T, D]

        # Projecao: [B, T, D] -> [B, T, 1]
        eidos_weights = self.weight_proj(balance_features)  # [B, T, 1]

        # Modula por confianca
        eidos_weights = eidos_weights * confidence.clamp(0.1, 1.0)

        return EidosOutput(
            eidos_weights=eidos_weights,
            axis_balance=axis_balance,
            axis_std=axis_std,
        )

    def extra_repr(self) -> str:
        return (
            f"drm_dim={self.drm_dim}, "
            f"base_decay={self.base_decay}, "
            f"base_reinforce={self.base_reinforce}, "
            f"dream_intensity={self.dream_intensity}"
        )
