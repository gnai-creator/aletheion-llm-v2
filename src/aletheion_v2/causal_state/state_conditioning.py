"""
StateConditioning + PolicyBinding: condicionamento causal do modelo.

Referencia ATIC: state/state_store.py + policy_binding.py.

StateConditioning: Projeta state_vector [B,4] para d_model e soma
condicionalmente ao hidden_states. ~25.5K params (d_model=768).

PolicyBinding: Heuristico (inference-only) que mapeia estado causal
para parametros de geracao (temperature, max_tokens). 0 params treinaveis.

State vector 4D:
  0 - curiosity [0, 1]: nivel de curiosidade
  1 - verbosity [0, 1]: nivel de verbosidade
  2 - exploration_depth [0, 1]: profundidade de exploracao
  3 - ask_followups [0, 1]: tendencia a fazer perguntas

Requer mudanca em model.py: aceitar state_vector opcional no forward.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict


STATE_DIM = 4
STATE_NAMES = ["curiosity", "verbosity", "exploration_depth", "ask_followups"]


@dataclass
class StateConditioningOutput:
    """Output do StateConditioning."""

    conditioned_hidden: torch.Tensor   # [B, T, d_model] hidden condicionado
    gate_value: torch.Tensor           # [B, 1] gate de aplicacao [0, 1]
    state_embedding: torch.Tensor      # [B, d_model] embedding do estado


@dataclass
class PolicyParams:
    """Parametros de policy inferidos do estado."""

    temperature: float
    max_tokens_factor: float  # Multiplicador sobre max_tokens base


class StateConditioning(nn.Module):
    """Condicionamento causal: injeta state_vector no hidden_states.

    Projeta vetor de estado 4D para d_model e soma ao hidden_states
    com gate aprendivel. Permite condicionar o comportamento do modelo
    em tempo de inferencia via estado causal externo.

    Args:
        d_model: dimensao do hidden state
        state_dim: dimensoes do estado causal (default 4)
        hidden_dim: dimensao interna do MLP (default 32)
    """

    def __init__(
        self,
        d_model: int = 768,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Projeta estado para d_model
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Gate de aplicacao: quanto do estado injetar
        self.gate = nn.Sequential(
            nn.Linear(state_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_vector: torch.Tensor,
    ) -> StateConditioningOutput:
        """Condiciona hidden_states com estado causal.

        Args:
            hidden_states: [B, T, d_model]
            state_vector: [B, 4] vetor de estado

        Returns:
            StateConditioningOutput
        """
        B, T, _ = hidden_states.shape

        # Projeta estado para d_model
        state_emb = self.state_embed(state_vector)  # [B, d_model]
        gate_value = self.gate(state_vector)          # [B, 1]

        # Expande para [B, T, d_model]
        state_expanded = state_emb.unsqueeze(1).expand(B, T, self.d_model)
        gate_expanded = gate_value.unsqueeze(1)  # [B, 1, 1]

        # Soma condicionada
        conditioned = hidden_states + gate_expanded * state_expanded

        return StateConditioningOutput(
            conditioned_hidden=conditioned,
            gate_value=gate_value,
            state_embedding=state_emb,
        )

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, state_dim={self.state_dim}"


class PolicyBinding:
    """Mapeamento heuristico de estado para parametros de geracao.

    Inference-only, sem parametros treinaveis.
    Mapeia cada dimensao do estado para parametros concretos:
    - verbosity -> max_tokens_factor
    - curiosity -> temperature
    """

    # Mapeamentos lineares
    VERBOSITY_TO_TOKENS = (0.4, 1.6)   # [0,1] -> [0.4x, 1.6x]
    CURIOSITY_TO_TEMP = (0.3, 0.9)     # [0,1] -> [0.3, 0.9]

    @staticmethod
    def bind(state_vector: torch.Tensor) -> PolicyParams:
        """Computa parametros de policy a partir do estado.

        Args:
            state_vector: [4] ou [B, 4] (usa media se batch)

        Returns:
            PolicyParams
        """
        if state_vector.dim() > 1:
            state_vector = state_vector.mean(dim=0)

        curiosity = state_vector[0].item()
        verbosity = state_vector[1].item()

        # Interpolacao linear
        temp_min, temp_max = PolicyBinding.CURIOSITY_TO_TEMP
        temperature = temp_min + curiosity * (temp_max - temp_min)

        tok_min, tok_max = PolicyBinding.VERBOSITY_TO_TOKENS
        max_tokens_factor = tok_min + verbosity * (tok_max - tok_min)

        return PolicyParams(
            temperature=temperature,
            max_tokens_factor=max_tokens_factor,
        )
