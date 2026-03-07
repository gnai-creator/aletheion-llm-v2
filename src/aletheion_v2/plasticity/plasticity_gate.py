"""
PlasticityGate: controle de plasticidade irreversivel ao longo da sequencia.

Referencia ATIC: irreversible/plasticity_gate.py.
Convertido para nn.Module com ~12.3K params (d_model=768).

A plasticidade comeca com um budget e se depleta ao longo da sequencia
conforme o modelo processa tokens de alto custo. O gate controla
quanta modificacao e permitida baseado na plasticidade restante.

Input: hidden_states [B,T,d_model], vi_severity [B,T,1]
Output: plasticity_remaining [B,T,1], gate_value [B,T,1]
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class PlasticityOutput:
    """Output do PlasticityGate."""

    plasticity_remaining: torch.Tensor  # [B, T, 1] plasticidade restante [0, 1]
    gate_value: torch.Tensor            # [B, T, 1] gate de controle [0, 1]
    cost_per_token: torch.Tensor        # [B, T, 1] custo estimado (diagnostico)


class PlasticityGate(nn.Module):
    """Gate de plasticidade com deplecao cumulativa.

    Estima custo de processamento por token, acumula ao longo de T,
    e produz um gate que limita modificacoes quando a plasticidade
    se esgota.

    Args:
        d_model: dimensao do hidden state
        initial_budget: budget inicial de plasticidade (default 1.0)
        depletion_rate: taxa de deplecao por custo (default 0.02)
    """

    def __init__(
        self,
        d_model: int = 768,
        initial_budget: float = 1.0,
        depletion_rate: float = 0.02,
    ):
        super().__init__()
        self.d_model = d_model
        self.initial_budget = initial_budget
        self.depletion_rate = depletion_rate

        # Estimador de custo: hidden -> custo positivo
        self.cost_estimator = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # Garante custo >= 0
        )

        # Gate: (plasticidade, severidade) -> gate [0, 1]
        self.gate_proj = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def _positional_budget(
        self, T: int, device: torch.device
    ) -> torch.Tensor:
        """Budget posicional que decai linearmente.

        De initial_budget para 0.7 * initial_budget ao longo de T.

        Returns:
            budget: [1, T, 1]
        """
        decay = torch.linspace(
            self.initial_budget,
            0.7 * self.initial_budget,
            T,
            device=device,
        )
        return decay.view(1, T, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vi_severity: torch.Tensor,
    ) -> PlasticityOutput:
        """Computa plasticidade e gate.

        Args:
            hidden_states: [B, T, d_model]
            vi_severity: [B, T, 1] severidade do VI

        Returns:
            PlasticityOutput
        """
        B, T, _ = hidden_states.shape

        # Estima custo por token
        cost = self.cost_estimator(hidden_states)  # [B, T, 1]

        # Budget posicional decrescente
        budget = self._positional_budget(T, hidden_states.device)  # [1, T, 1]

        # Plasticidade restante: budget - deplecao cumulativa
        cumulative_cost = torch.cumsum(cost, dim=1) * self.depletion_rate
        plasticity = (budget - cumulative_cost).clamp(0.0, 1.0)  # [B, T, 1]

        # Gate: combina plasticidade com severidade
        gate_input = torch.cat([plasticity, vi_severity], dim=-1)  # [B, T, 2]
        gate = self.gate_proj(gate_input)  # [B, T, 1]

        return PlasticityOutput(
            plasticity_remaining=plasticity,
            gate_value=gate,
            cost_per_token=cost,
        )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"initial_budget={self.initial_budget}, "
            f"depletion_rate={self.depletion_rate}"
        )
