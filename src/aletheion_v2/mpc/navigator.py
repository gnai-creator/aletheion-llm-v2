"""
Manifold Navigator: Beam search MPC para navegacao no manifold.

Planeja sequencia de acoes para maximizar phi(M) enquanto
minimiza custo de intervencao. Opera em inferencia-only
(acoes discretas nao sao diferenciaveisno treinamento).

Modos:
- RECOVERY: phi < phi_floor -> maximiza correcao
- MAINTENANCE: phi >= phi_floor -> minimiza intervencao

Beam search com K=4 (width) e D=3 (depth).
"""

import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass

from aletheion_v2.mpc.transition_model import (
    TransitionModel, InterventionType, INTERVENTION_COSTS, NUM_ACTIONS,
)


@dataclass
class NavigationPlan:
    """Plano de navegacao produzido pelo beam search."""
    actions: List[int]         # Sequencia de acoes
    predicted_phi: List[float] # phi predito apos cada acao
    total_cost: float          # Custo total de intervencao
    mode: str                  # "recovery" ou "maintenance"


@dataclass
class NavigationState:
    """Estado atual da navegacao (para dashboard)."""
    current_phi: float
    mode: str
    last_action: int
    plan: Optional[NavigationPlan]
    oscillation_index: float


class ManifoldNavigator:
    """Navegador MPC com beam search.

    Planeja K*D sequencias de acoes e escolhe a melhor
    baseado em phi futuro e custo de intervencao.

    Args:
        transition_model: Modelo de transicao T(s, a) -> s'
        beam_width: Largura do beam search (K)
        lookahead_depth: Profundidade do lookahead (D)
        phi_floor: Limiar minimo de phi
        intervention_cost_weight: Peso do custo de intervencao
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        beam_width: int = 4,
        lookahead_depth: int = 3,
        phi_floor: float = 0.45,
        intervention_cost_weight: float = 0.20,
    ):
        self.transition_model = transition_model
        self.beam_width = beam_width
        self.lookahead_depth = lookahead_depth
        self.phi_floor = phi_floor
        self.cost_weight = intervention_cost_weight

        # Estado
        self.last_action = InterventionType.NOOP
        self.action_history: List[int] = []

    def _get_mode(self, phi_total: float) -> str:
        """Determina modo (recovery ou maintenance)."""
        return "recovery" if phi_total < self.phi_floor else "maintenance"

    def _compute_score(
        self,
        phi_sequence: List[float],
        actions: List[int],
        mode: str,
    ) -> float:
        """Computa score de uma sequencia de acoes.

        Recovery: maximiza phi final
        Maintenance: minimiza custo (com bonus por estabilidade)
        """
        if not phi_sequence:
            return 0.0

        phi_final = phi_sequence[-1]
        total_cost = sum(
            INTERVENTION_COSTS.get(InterventionType(a), 0.0) for a in actions
        )

        # Smoothness: penaliza mudancas de acao
        smoothness_penalty = 0.0
        prev = self.last_action
        for a in actions:
            if a != prev:
                smoothness_penalty += 0.02
            prev = a

        if mode == "recovery":
            # Maximiza phi, aceita custo
            score = phi_final - self.cost_weight * total_cost - smoothness_penalty
        else:
            # Minimiza custo, mantem phi
            phi_bonus = max(0, phi_final - self.phi_floor) * 0.5
            score = phi_bonus - total_cost - smoothness_penalty

        return score

    @torch.no_grad()
    def plan(self, phi_components: torch.Tensor) -> NavigationPlan:
        """Executa beam search para encontrar melhor plano.

        Args:
            phi_components: [4] phi_components atuais

        Returns:
            NavigationPlan com melhor sequencia de acoes
        """
        # Pesos do phi_total
        weights = torch.tensor([0.35, 0.25, 0.25, 0.15])
        phi_total = (phi_components * weights).sum().item()
        mode = self._get_mode(phi_total)

        # Beam: lista de (state, actions, phi_sequence, score)
        state = phi_components.unsqueeze(0)  # [1, 4]

        # Inicializa beams
        beams = [(state, [], [phi_total])]

        for depth in range(self.lookahead_depth):
            candidates = []

            for beam_state, beam_actions, beam_phis in beams:
                for action_idx in range(NUM_ACTIONS):
                    # One-hot
                    action_oh = torch.zeros(1, NUM_ACTIONS)
                    action_oh[0, action_idx] = 1.0

                    # Prediz proximo estado
                    next_state = self.transition_model(
                        beam_state, action_oh.to(beam_state.device)
                    )

                    # phi_total do proximo estado
                    next_phi = (next_state[0] * weights.to(next_state.device)).sum().item()

                    new_actions = beam_actions + [action_idx]
                    new_phis = beam_phis + [next_phi]

                    score = self._compute_score(new_phis, new_actions, mode)
                    candidates.append((next_state, new_actions, new_phis, score))

            # Seleciona top-K
            candidates.sort(key=lambda x: x[3], reverse=True)
            beams = [
                (c[0], c[1], c[2]) for c in candidates[: self.beam_width]
            ]

        # Melhor beam
        best_state, best_actions, best_phis = beams[0]
        total_cost = sum(
            INTERVENTION_COSTS.get(InterventionType(a), 0.0)
            for a in best_actions
        )

        plan = NavigationPlan(
            actions=best_actions,
            predicted_phi=best_phis,
            total_cost=total_cost,
            mode=mode,
        )

        # Atualiza estado
        if best_actions:
            self.last_action = best_actions[0]
            self.action_history.append(best_actions[0])

        return plan

    def get_state(self, phi_total: float) -> NavigationState:
        """Retorna estado atual para dashboard."""
        return NavigationState(
            current_phi=phi_total,
            mode=self._get_mode(phi_total),
            last_action=self.last_action,
            plan=None,
            oscillation_index=self._oscillation_index(),
        )

    def _oscillation_index(self) -> float:
        """Mede oscilacao recente nas acoes."""
        if len(self.action_history) < 3:
            return 0.0
        recent = self.action_history[-10:]
        changes = sum(
            1 for i in range(1, len(recent)) if recent[i] != recent[i - 1]
        )
        return changes / max(len(recent) - 1, 1)
