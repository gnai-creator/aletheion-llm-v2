"""
AmbiguityHead: deteccao de nivel e tipo de ambiguidade por token.

Referencia ATIC: grounding/ambiguity_detector.py (deterministico).
Convertido para nn.Module com ~49.5K params (d_model=768).

5 AmbiguityTypes:
  0 - PRONOUN: ambiguidade pronominal
  1 - LEXICAL: ambiguidade lexica (polissemia)
  2 - STRUCTURAL: ambiguidade estrutural/sintatica
  3 - SCOPE: ambiguidade de escopo
  4 - REFERENTIAL: ambiguidade referencial

Input: hidden_states [B, T, d_model]
Output: ambiguity_level [B, T, 1], ambiguity_type [B, T, 5]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


NUM_AMBIGUITY_TYPES = 5

AMBIGUITY_TYPE_NAMES = [
    "PRONOUN", "LEXICAL", "STRUCTURAL", "SCOPE", "REFERENTIAL",
]


@dataclass
class AmbiguityOutput:
    """Output do AmbiguityHead."""

    ambiguity_level: torch.Tensor  # [B, T, 1] nivel de ambiguidade [0, 1]
    ambiguity_type: torch.Tensor   # [B, T, 5] probabilidades por tipo


class AmbiguityHead(nn.Module):
    """Detecta nivel e tipo de ambiguidade por token.

    Args:
        d_model: dimensao do hidden state
        hidden_dim: dimensao interna do MLP (default 32)
        num_types: numero de tipos de ambiguidade (default 5)
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 32,
        num_types: int = NUM_AMBIGUITY_TYPES,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_types = num_types

        # MLP para nivel de ambiguidade
        self.level_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # MLP para tipo de ambiguidade
        self.type_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_types),
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> AmbiguityOutput:
        """Detecta ambiguidade.

        Args:
            hidden_states: [B, T, d_model]

        Returns:
            AmbiguityOutput com level e type
        """
        level = self.level_net(hidden_states)      # [B, T, 1]
        type_logits = self.type_net(hidden_states)  # [B, T, num_types]
        type_probs = F.softmax(type_logits, dim=-1)

        return AmbiguityOutput(
            ambiguity_level=level,
            ambiguity_type=type_probs,
        )

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_types={self.num_types}"
