"""
Transition Model: T(s, a) -> s' (aprendido).

Modelo de transicao que prediz proximo estado do manifold
dado estado atual e acao. Usado pelo MPC navigator.

12 acoes disponiveis:
    NOOP, INJECT_AXIS_0..4, INJECT_WEAKEST, INJECT_TWO,
    CONF_LIGHT, CONF_MEDIUM, CONF_STRONG, RESET_25, RESET_50

Estado: phi_components [4] (dim, disp, ent, conf)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from enum import IntEnum


class InterventionType(IntEnum):
    """Tipos de intervencao no manifold."""
    NOOP = 0
    INJECT_AXIS_0 = 1
    INJECT_AXIS_1 = 2
    INJECT_AXIS_2 = 3
    INJECT_AXIS_3 = 4
    INJECT_AXIS_4 = 5
    INJECT_WEAKEST = 6
    INJECT_TWO = 7
    CONF_LIGHT = 8
    CONF_MEDIUM = 9
    CONF_STRONG = 10
    RESET_25 = 11


# Custos de intervencao (heuristicos)
INTERVENTION_COSTS = {
    InterventionType.NOOP: 0.00,
    InterventionType.INJECT_AXIS_0: 0.04,
    InterventionType.INJECT_AXIS_1: 0.04,
    InterventionType.INJECT_AXIS_2: 0.04,
    InterventionType.INJECT_AXIS_3: 0.04,
    InterventionType.INJECT_AXIS_4: 0.04,
    InterventionType.INJECT_WEAKEST: 0.06,
    InterventionType.INJECT_TWO: 0.08,
    InterventionType.CONF_LIGHT: 0.03,
    InterventionType.CONF_MEDIUM: 0.06,
    InterventionType.CONF_STRONG: 0.10,
    InterventionType.RESET_25: 0.15,
}

NUM_ACTIONS = len(InterventionType)


class TransitionModel(nn.Module):
    """Modelo de transicao aprendido T(s, a) -> s'.

    Recebe estado (phi_components) e acao (one-hot),
    prediz proximo estado do manifold.

    Combina componente analitico (regras fixas) com
    componente neural (residuo aprendido).

    Args:
        state_dim: Dimensao do estado (4 = phi_components)
        num_actions: Numero de acoes (12)
        hidden_dim: Dimensao oculta
    """

    def __init__(
        self,
        state_dim: int = 4,
        num_actions: int = NUM_ACTIONS,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Rede neural para residuo
        self.net = nn.Sequential(
            nn.Linear(state_dim + num_actions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh(),  # Residuo em [-1, 1]
        )

        # Escala do residuo (aprendida)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _analytical_transition(
        self,
        state: torch.Tensor,
        action: int,
    ) -> torch.Tensor:
        """Transicao analitica (regras fixas).

        Args:
            state: [B, 4] phi_components
            action: indice da acao

        Returns:
            next_state: [B, 4]
        """
        next_state = state.clone()
        a = InterventionType(action)

        if a == InterventionType.NOOP:
            next_state = next_state * 0.995  # Drift natural

        elif InterventionType.INJECT_AXIS_0 <= a <= InterventionType.INJECT_AXIS_4:
            # Injecao de eixo especifico: melhora phi_dim e phi_ent
            next_state[:, 0] = (next_state[:, 0] + 0.05).clamp(0, 1)
            next_state[:, 2] = (next_state[:, 2] + 0.03).clamp(0, 1)

        elif a == InterventionType.INJECT_WEAKEST:
            next_state[:, 0] = (next_state[:, 0] + 0.07).clamp(0, 1)
            next_state[:, 2] = (next_state[:, 2] + 0.05).clamp(0, 1)

        elif a == InterventionType.INJECT_TWO:
            next_state[:, 0] = (next_state[:, 0] + 0.10).clamp(0, 1)
            next_state[:, 2] = (next_state[:, 2] + 0.07).clamp(0, 1)

        elif a == InterventionType.CONF_LIGHT:
            factor = 0.05
            next_state[:, 3] = next_state[:, 3] + factor * (1 - next_state[:, 3])

        elif a == InterventionType.CONF_MEDIUM:
            factor = 0.10
            next_state[:, 3] = next_state[:, 3] + factor * (1 - next_state[:, 3])

        elif a == InterventionType.CONF_STRONG:
            factor = 0.20
            next_state[:, 3] = next_state[:, 3] + factor * (1 - next_state[:, 3])

        elif a == InterventionType.RESET_25:
            next_state[:, 0] = (next_state[:, 0] + 0.10).clamp(0, 1)
            next_state[:, 2] = (next_state[:, 2] + 0.10).clamp(0, 1)

        return next_state

    def forward(
        self,
        state: torch.Tensor,
        action_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Prediz proximo estado.

        Combina transicao analitica com residuo neural.

        Args:
            state: [B, state_dim] phi_components
            action_onehot: [B, num_actions] acao em one-hot

        Returns:
            next_state: [B, state_dim]
        """
        # Componente neural (residuo)
        net_input = torch.cat([state, action_onehot], dim=-1)
        residual = self.net(net_input) * self.residual_scale

        # Componente analitico (acao com maior peso)
        action_idx = action_onehot.argmax(dim=-1)  # [B]
        # Usa a primeira acao do batch (simplificacao para beam search)
        analytical = self._analytical_transition(state, action_idx[0].item())

        # Combina
        next_state = (analytical + residual).clamp(0, 1)
        return next_state
