"""
DensityTracker: Hash-grid esparso 5D para estimar densidade no manifold.

Referencia ATIC: drm/mpl_density.py.
NAO e nn.Module - e um utility sem parametros treinaveis.

Usa grid esparso (dict) com kernel Gaussiano para estimar
densidade local. Atualizado pelo trainer apos cada batch.

Toda operacao e @torch.no_grad() para eficiencia.
"""

import torch
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DensityInfo:
    """Informacao de densidade para um conjunto de coordenadas."""

    density: torch.Tensor   # [B, T, 1] densidade local estimada [0, 1]
    novelty: torch.Tensor   # [B, T, 1] novidade = 1 - density


class DensityTracker:
    """Hash-grid esparso 5D para estimativa de densidade.

    Discretiza coordenadas em celulas de grid e acumula contagens.
    Densidade e computada via kernel Gaussiano usando celulas vizinhas.

    Args:
        resolution: numero de bins por eixo (default 10)
        bandwidth: largura do kernel Gaussiano (default 0.15)
        drm_dim: dimensoes do manifold (default 5)
    """

    def __init__(
        self,
        resolution: int = 10,
        bandwidth: float = 0.15,
        drm_dim: int = 5,
    ):
        self.resolution = resolution
        self.bandwidth = bandwidth
        self.drm_dim = drm_dim

        # Grid esparso: chave = tuple de indices, valor = contagem
        self._grid: Dict[Tuple[int, ...], float] = {}
        self._total_updates = 0

    def _coords_to_cell(self, coords: torch.Tensor) -> torch.Tensor:
        """Converte coordenadas [0,1] em indices de grid.

        Args:
            coords: [..., D] coordenadas

        Returns:
            cells: [..., D] indices inteiros
        """
        return (coords.clamp(0, 1 - 1e-6) * self.resolution).long()

    @torch.no_grad()
    def update(self, coords: torch.Tensor) -> None:
        """Atualiza grid com novas coordenadas.

        Args:
            coords: [B, T, D] ou [N, D] coordenadas
        """
        flat = coords.detach().view(-1, self.drm_dim)
        cells = self._coords_to_cell(flat)

        for i in range(cells.shape[0]):
            key = tuple(cells[i].tolist())
            self._grid[key] = self._grid.get(key, 0.0) + 1.0

        self._total_updates += flat.shape[0]

    @torch.no_grad()
    def query(self, coords: torch.Tensor) -> DensityInfo:
        """Consulta densidade para coordenadas.

        Args:
            coords: [B, T, D] coordenadas

        Returns:
            DensityInfo com density e novelty
        """
        original_shape = coords.shape[:-1]
        flat = coords.detach().view(-1, self.drm_dim)
        cells = self._coords_to_cell(flat)

        densities = torch.zeros(flat.shape[0], device=coords.device)

        if self._total_updates > 0:
            for i in range(flat.shape[0]):
                key = tuple(cells[i].tolist())
                count = self._grid.get(key, 0.0)
                # Kernel Gaussiano: soma vizinhos imediatos
                for d in range(self.drm_dim):
                    key_minus = list(key)
                    key_minus[d] = max(0, key[d] - 1)
                    key_plus = list(key)
                    key_plus[d] = min(self.resolution - 1, key[d] + 1)
                    count += self._grid.get(tuple(key_minus), 0.0) * math.exp(
                        -1.0 / (2 * self.bandwidth ** 2)
                    )
                    count += self._grid.get(tuple(key_plus), 0.0) * math.exp(
                        -1.0 / (2 * self.bandwidth ** 2)
                    )

                densities[i] = count / self._total_updates

        # Normaliza para [0, 1]
        d_max = densities.max()
        if d_max > 0:
            densities = densities / d_max

        density = densities.view(*original_shape, 1)
        novelty = 1.0 - density

        return DensityInfo(density=density, novelty=novelty)

    def reset(self) -> None:
        """Limpa o grid."""
        self._grid.clear()
        self._total_updates = 0

    @property
    def num_cells(self) -> int:
        """Numero de celulas ocupadas."""
        return len(self._grid)
