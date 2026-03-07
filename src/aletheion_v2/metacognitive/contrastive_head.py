"""
ContrastiveHead: Auto-avaliacao contrastiva de representacoes.

Referencia ATIC: metacognitive/adversarial_engine.py + injection_engine.py.
Convertido para nn.Module com ~200K params (d_model=768, hidden=128).

Cria duas visoes (projecoes) dos hidden_states e mede divergencia.
Alta divergencia indica instabilidade na representacao.
Baixa divergencia indica representacao consistente/robusta.

Input: hidden_states [B, T, d_model]
Output: divergence [B, T, 1], view_a [B, T, proj_dim], view_b [B, T, proj_dim]
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContrastiveOutput:
    """Output do ContrastiveHead."""

    divergence: torch.Tensor    # [B, T, 1] divergencia entre visoes [0, 1]
    view_a: torch.Tensor        # [B, T, proj_dim] projecao A
    view_b: torch.Tensor        # [B, T, proj_dim] projecao B


class ContrastiveHead(nn.Module):
    """Auto-avaliacao contrastiva via projecoes duais.

    Projeta hidden_states por dois caminhos independentes e mede
    divergencia. Serve como sinal metacognitivo: se as projecoes
    divergem muito, o modelo esta incerto sobre a representacao.

    Args:
        d_model: dimensao do hidden state
        hidden_dim: dimensao interna dos projetores (default 128)
        proj_dim: dimensao da projecao final (default 384)
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 128,
        proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim or (d_model // 2)

        # Projetor A
        self.proj_a = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.proj_dim),
        )

        # Projetor B (arquitetura identica, pesos diferentes)
        self.proj_b = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.proj_dim),
        )

        # MLP de divergencia: concat projecoes -> scalar
        self.divergence_net = nn.Sequential(
            nn.Linear(self.proj_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> ContrastiveOutput:
        """Computa divergencia contrastiva.

        Args:
            hidden_states: [B, T, d_model]

        Returns:
            ContrastiveOutput
        """
        # Projecoes independentes
        view_a = self.proj_a(hidden_states)  # [B, T, proj_dim]
        view_b = self.proj_b(hidden_states)  # [B, T, proj_dim]

        # Divergencia
        concat = torch.cat([view_a, view_b], dim=-1)  # [B, T, 2*proj_dim]
        divergence = self.divergence_net(concat)        # [B, T, 1]

        return ContrastiveOutput(
            divergence=divergence,
            view_a=view_a,
            view_b=view_b,
        )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"hidden_dim={self.hidden_dim}, "
            f"proj_dim={self.proj_dim}"
        )
