"""
Gates Q1/Q2: Incerteza aleatoria e epistemica.

Portado do epistemic_softmax.py para formato compativel com
o pipeline do AletheionV2. Opera sobre hidden_states [B, T, d_model].

Formulas:
    q1 = Q1Gate(hidden)  -- incerteza aleatoria [0, 1]
    q2 = Q2Gate(hidden)  -- incerteza epistemica [0, 1]
    c = q1 * q2          -- confianca combinada
    tau = tau0 / c        -- temperatura adaptativa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GateNetwork(nn.Module):
    """MLP que produz valor escalar em [0, 1] (sigmoid) ou [0, inf) (softplus).

    Usado como building block para Q1Gate e Q2Gate.

    Args:
        input_dim: Dimensao do input (d_model)
        hidden_dim: Dimensao das camadas ocultas
        num_layers: Numero de camadas ocultas
        dropout: Probabilidade de dropout
        output_activation: 'sigmoid' para gates, 'softplus' para variancias
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_activation: str = "sigmoid",
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "softplus":
            layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, input_dim] ou [B, input_dim]

        Returns:
            gate_value: [B, T, 1] ou [B, 1]
        """
        return self.network(x)


class Q1Gate(nn.Module):
    """Gate de incerteza aleatoria (Q1).

    Captura incerteza irredutivel inerente aos dados.
    Q1 alto = dados ambiguos, Q1 baixo = dados claros.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gate = GateNetwork(d_model, hidden_dim, num_layers, dropout, "sigmoid")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Computa incerteza aleatoria.

        Args:
            hidden_states: [B, T, d_model]

        Returns:
            q1: [B, T, 1] em [0, 1]
        """
        return self.gate(hidden_states)


class Q2Gate(nn.Module):
    """Gate de incerteza epistemica (Q2).

    Captura incerteza redutivel por falta de conhecimento.
    Q2 alto = modelo incerto, Q2 baixo = modelo confiante.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gate = GateNetwork(d_model, hidden_dim, num_layers, dropout, "sigmoid")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Computa incerteza epistemica.

        Args:
            hidden_states: [B, T, d_model]

        Returns:
            q2: [B, T, 1] em [0, 1]
        """
        return self.gate(hidden_states)


class AdaptiveTemperature(nn.Module):
    """Temperatura adaptativa baseada em confianca Q1*Q2.

    Formula:
        c = clamp(q1 * q2, eps, 1)
        tau = tau0 / c   se c < threshold
        tau = tau0        caso contrario
    """

    def __init__(
        self,
        base_temperature: float = 1.0,
        tau_threshold: float = 0.5,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.base_temperature = base_temperature
        self.tau_threshold = tau_threshold
        self.epsilon = epsilon

    def forward(
        self, q1: torch.Tensor, q2: torch.Tensor
    ) -> torch.Tensor:
        """Computa temperatura adaptativa.

        Args:
            q1: [B, T, 1] incerteza aleatoria
            q2: [B, T, 1] incerteza epistemica

        Returns:
            temperature: [B, T, 1] temperatura por token
        """
        c = torch.clamp(q1 * q2, min=self.epsilon, max=1.0)
        tau = torch.where(
            c < self.tau_threshold,
            self.base_temperature / c,
            torch.full_like(c, self.base_temperature),
        )
        return tau
