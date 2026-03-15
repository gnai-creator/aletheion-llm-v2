"""
GravityField: Campo de feedback acumulado sobre o manifold epistemico.

Acumula sinais de feedback humano como um campo sobre o manifold 5D.
Feedback positivo diminui gravidade (caminho fica mais barato).
Feedback negativo aumenta gravidade (caminho fica mais caro).

NAO ativo durante treino inicial -- gravity_field defaults para zeros,
tornando gravitational_objective identico ao real_geodesic ate que
dados de feedback estejam disponiveis.

Notas de design:
- Campo suavizado com decay temporal para evitar oscilacao
- Regioes cold-start tem gravidade zero (prior neutro)
- Opera no mesmo espaco 5D do manifold epistemico
"""

import torch
import torch.nn as nn


class GravityField(nn.Module):
    """Campo gravitacional acumulado sobre o manifold epistemico.

    Args:
        dim: Dimensionalidade do manifold (deve coincidir com MetricNet dim)
        decay: Fator de decay temporal para suavizacao (0.99 = decay lento)
    """

    def __init__(self, dim: int = 5, decay: float = 0.99):
        super().__init__()
        self.dim = dim
        self.decay = decay

        # Campo acumulado -- comeca em zero (neutro)
        self.register_buffer("accumulated_field", torch.zeros(dim))

    def update(
        self,
        coords: torch.Tensor,
        feedback_signal: float,
    ) -> None:
        """Atualiza campo gravitacional nas coordenadas dadas.

        Args:
            coords: [dim] posicao no manifold epistemico
            feedback_signal: float em [-1, 1]
                positivo = aprovacao humana (diminui gravidade)
                negativo = rejeicao humana (aumenta gravidade)
        """
        self.accumulated_field = (
            self.decay * self.accumulated_field
            + (1.0 - self.decay) * feedback_signal * coords.detach()
        )

    def get_field(self, coords: torch.Tensor) -> torch.Tensor:
        """Retorna campo gravitacional nas coordenadas dadas.

        Atualmente retorna campo global acumulado (independente de posicao).
        Extensao futura: interpolar campo baseado em distancia aos pontos
        de feedback.

        Args:
            coords: [..., dim]

        Returns:
            gravity_field: [..., dim]
        """
        return self.accumulated_field.expand_as(coords)

    def reset(self) -> None:
        """Reseta campo para estado neutro."""
        self.accumulated_field.zero_()

    def get_state_dict_gravity(self) -> dict:
        """Estado para checkpoint/diagnostico."""
        return {
            "accumulated_field": self.accumulated_field.clone(),
            "decay": self.decay,
            "field_norm": self.accumulated_field.norm().item(),
        }
