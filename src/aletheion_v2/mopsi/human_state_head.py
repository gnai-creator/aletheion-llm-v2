"""
HumanStateHead + PhiPsiMediator: estimativa de estado humano e mediacao.

Referencia ATIC: drm/mopsi_*.py (~1000 linhas).
Convertido para nn.Module com ~25.3K params (HumanState) + ~41 params (Mediator).

HumanStateHead: estima estado do interlocutor humano em 5D
e computa psi (satisfacao/beneficio para o humano).

PhiPsiMediator: media trade-off entre phi (saude epistemica)
e psi (satisfacao humana) usando conflito como moderador.

Depende de: Filosofia3 (conflict_intensity)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class HumanStateOutput:
    """Output do HumanStateHead."""

    human_state: torch.Tensor  # [B, T, 5] estado humano estimado
    psi: torch.Tensor          # [B, T, 1] satisfacao humana [0, 1]


@dataclass
class MediationOutput:
    """Output do PhiPsiMediator."""

    mediated_score: torch.Tensor  # [B, T, 1] score mediado [0, 1]
    phi_weight: torch.Tensor      # [B, T, 1] peso dado a phi (diagnostico)


class HumanStateHead(nn.Module):
    """Estima estado do interlocutor humano e satisfacao.

    O estado humano e uma representacao 5D aprendivel que captura
    aspectos do contexto relevantes para o interlocutor.
    Psi e um scalar que resume a satisfacao/beneficio estimado.

    Args:
        d_model: dimensao do hidden state
        hidden_dim: dimensao interna (default 32)
        state_dim: dimensoes do estado humano (default 5)
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 32,
        state_dim: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # MLP: hidden_states -> estado humano 5D
        self.state_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid(),
        )

        # MLP: (estado_humano[5] + phi_components[4] + confidence[1]) -> psi
        self.psi_net = nn.Sequential(
            nn.Linear(state_dim + 4 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        phi_components: torch.Tensor,
        confidence: torch.Tensor,
    ) -> HumanStateOutput:
        """Estima estado humano e psi.

        Args:
            hidden_states: [B, T, d_model]
            phi_components: [B, T, 4]
            confidence: [B, T, 1]

        Returns:
            HumanStateOutput
        """
        # Estado humano estimado
        human_state = self.state_net(hidden_states)  # [B, T, 5]

        # Psi: combina estado + phi + confianca
        psi_input = torch.cat([
            human_state, phi_components, confidence,
        ], dim=-1)  # [B, T, 10]
        psi = self.psi_net(psi_input)  # [B, T, 1]

        return HumanStateOutput(
            human_state=human_state,
            psi=psi,
        )

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, state_dim={self.state_dim}"


class PhiPsiMediator(nn.Module):
    """Media trade-off entre phi e psi.

    Produz um score mediado que balanceia saude epistemica (phi)
    com satisfacao humana (psi), usando conflito como moderador.

    Quando conflito e alto, o mediator tende a favorecer phi
    (prioriza integridade epistemica).

    Args:
        Nenhum argumento de config - modulo simples.
    """

    def __init__(self):
        super().__init__()

        # MLP: (phi, psi, conflict) -> score mediado
        self.mediator = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        phi_total: torch.Tensor,
        psi: torch.Tensor,
        conflict_intensity: torch.Tensor,
    ) -> MediationOutput:
        """Media phi e psi.

        Args:
            phi_total: [B, T, 1] saude epistemica
            psi: [B, T, 1] satisfacao humana
            conflict_intensity: [B, T, 1] intensidade de conflito

        Returns:
            MediationOutput
        """
        # Concat: [B, T, 3]
        mediator_input = torch.cat([
            phi_total, psi, conflict_intensity,
        ], dim=-1)

        mediated = self.mediator(mediator_input)  # [B, T, 1]

        # Peso implicito de phi: quanto mais conflito, mais phi
        # (heuristica de diagnostico)
        phi_weight = 0.5 + 0.3 * conflict_intensity

        return MediationOutput(
            mediated_score=mediated,
            phi_weight=phi_weight,
        )
