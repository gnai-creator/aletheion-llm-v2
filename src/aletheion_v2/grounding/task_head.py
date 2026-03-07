"""
TaskClassificationHead: classificacao de tipo de tarefa por token.

Referencia ATIC: grounding/task_classifier.py (deterministico, 8 tipos).
Convertido para nn.Module com ~49.9K params (d_model=768).

9 TaskTypes:
  0 - FACTUAL: fatos objetivos
  1 - ANALYTICAL: analise e raciocinio
  2 - CREATIVE: criacao de conteudo
  3 - CONVERSATIONAL: dialogo casual
  4 - INSTRUCTIONAL: instrucoes/how-to
  5 - OPINION: opiniao subjetiva
  6 - CODE: programacao
  7 - MATHEMATICAL: matematica/logica
  8 - EPISTEMIC: questoes sobre conhecimento/incerteza

Input: hidden_states [B, T, d_model]
Output: task_probs [B, T, 9], task_confidence [B, T, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


NUM_TASK_TYPES = 9

TASK_TYPE_NAMES = [
    "FACTUAL", "ANALYTICAL", "CREATIVE", "CONVERSATIONAL",
    "INSTRUCTIONAL", "OPINION", "CODE", "MATHEMATICAL", "EPISTEMIC",
]


@dataclass
class TaskOutput:
    """Output do TaskClassificationHead."""

    task_probs: torch.Tensor       # [B, T, 9] probabilidades por tipo
    task_confidence: torch.Tensor  # [B, T, 1] confianca na classificacao


class TaskClassificationHead(nn.Module):
    """Classifica o tipo de tarefa de cada token.

    Args:
        d_model: dimensao do hidden state
        hidden_dim: dimensao interna do MLP (default 64)
        num_types: numero de tipos de tarefa (default 9)
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 64,
        num_types: int = NUM_TASK_TYPES,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_types = num_types

        # MLP: d_model -> hidden -> num_types
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_types),
        )

        # Projecao de confianca: logits -> scalar
        self.conf_proj = nn.Sequential(
            nn.Linear(num_types, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> TaskOutput:
        """Classifica tipo de tarefa.

        Args:
            hidden_states: [B, T, d_model]

        Returns:
            TaskOutput com probs e confidence
        """
        logits = self.classifier(hidden_states)  # [B, T, num_types]
        task_probs = F.softmax(logits, dim=-1)    # [B, T, num_types]
        task_confidence = self.conf_proj(logits)   # [B, T, 1]

        return TaskOutput(
            task_probs=task_probs,
            task_confidence=task_confidence,
        )

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_types={self.num_types}"
