"""
Phi Field: phi(M) - saude global do manifold epistemico.

Computa 4 componentes de saude a partir de batch statistics
(sem historico, totalmente diferenciavel):

    phi_dim:  diversidade dimensional (std das coords)
    phi_disp: dispersao dos pontos (distancia media ao centroide)
    phi_ent:  entropia de distribuicao nos eixos
    phi_conf: variancia da confianca (U-shape: ideal_std = 0.15)

phi_total = w_dim*phi_dim + w_disp*phi_disp + w_ent*phi_ent + w_conf*phi_conf
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PhiField(nn.Module):
    """Computa phi(M) - saude do manifold - a partir de batch statistics.

    Opera sobre coordenadas e confiancas do batch atual.
    Nao requer historico (diferenciavel end-to-end).

    Args:
        drm_dim: Dimensao do manifold (5)
        weights: Pesos (phi_dim, phi_disp, phi_ent, phi_conf)
        ideal_conf_std: Desvio padrao ideal da confianca
        eps: Epsilon para estabilidade numerica
    """

    def __init__(
        self,
        drm_dim: int = 5,
        weights: tuple = (0.35, 0.25, 0.25, 0.15),
        ideal_conf_std: float = 0.15,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.drm_dim = drm_dim
        self.ideal_conf_std = ideal_conf_std
        self.eps = eps

        # Pesos como parametros (treinaveis opcionalmente)
        self.register_buffer(
            "weights", torch.tensor(weights, dtype=torch.float32)
        )

    def _phi_dim(self, coords: torch.Tensor) -> torch.Tensor:
        """Diversidade dimensional.

        Mede quao espalhados estao os pontos em cada eixo.
        Alta variancia por eixo = alta diversidade.

        Args:
            coords: [B, T, D]

        Returns:
            phi_dim: [B, T, 1]
        """
        # Variancia por eixo no batch
        # [B, T, D] -> media por eixo: [B, D]
        axis_std = coords.std(dim=1, keepdim=True)  # [B, 1, D]
        # Media dos stds normalizados (max std em [0,1] = ~0.29)
        phi_dim = (axis_std.mean(dim=-1, keepdim=True) / 0.29).clamp(0, 1)
        # Broadcast para [B, T, 1]
        phi_dim = phi_dim.expand(-1, coords.shape[1], -1)
        return phi_dim

    def _phi_disp(self, coords: torch.Tensor) -> torch.Tensor:
        """Dispersao dos pontos (distancia media ao centroide do batch).

        Args:
            coords: [B, T, D]

        Returns:
            phi_disp: [B, T, 1]
        """
        centroid = coords.mean(dim=1, keepdim=True)  # [B, 1, D]
        dist = torch.norm(coords - centroid, dim=-1, keepdim=True)  # [B, T, 1]
        # Normaliza: max dist em [0,1]^5 = sqrt(5) ~ 2.24
        phi_disp = (dist / 2.24).clamp(0, 1)
        return phi_disp

    def _phi_ent(self, coords: torch.Tensor) -> torch.Tensor:
        """Entropia de distribuicao nos eixos.

        Mede se todos os eixos sao igualmente utilizados.
        Maxima quando distribuicao e uniforme.

        Args:
            coords: [B, T, D]

        Returns:
            phi_ent: [B, T, 1]
        """
        # "Frequencia" de cada eixo: media dos valores
        axis_freq = coords.mean(dim=1)  # [B, D]
        # Normaliza para distribuicao
        axis_freq = axis_freq / (axis_freq.sum(dim=-1, keepdim=True) + self.eps)
        # Entropia normalizada
        log_freq = torch.log(axis_freq + self.eps)
        entropy = -(axis_freq * log_freq).sum(dim=-1, keepdim=True)  # [B, 1]
        max_entropy = math.log(self.drm_dim)
        phi_ent = (entropy / max_entropy).clamp(0, 1)
        # Broadcast para [B, T, 1]
        phi_ent = phi_ent.unsqueeze(1).expand(-1, coords.shape[1], -1)
        return phi_ent

    def _phi_conf(self, confidence: torch.Tensor) -> torch.Tensor:
        """Variancia da confianca (U-shape com ideal_std).

        Ideal: std da confianca proximo de ideal_conf_std.
        Muito baixo = confianca uniforme (ruim).
        Muito alto = confianca erratica (ruim).

        Args:
            confidence: [B, T, 1]

        Returns:
            phi_conf: [B, T, 1]
        """
        conf_std = confidence.std(dim=1, keepdim=True)  # [B, 1, 1]
        # U-shape: maximo quando std = ideal_std
        deviation = (conf_std - self.ideal_conf_std).abs()
        phi_conf = (1.0 - deviation / 0.5).clamp(0, 1)
        phi_conf = phi_conf.expand(-1, confidence.shape[1], -1)
        return phi_conf

    def forward(
        self,
        coords: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computa phi(M) completo.

        Args:
            coords: [B, T, drm_dim] coordenadas no manifold
            confidence: [B, T, 1] confianca MAD

        Returns:
            phi_components: [B, T, 4] (dim, disp, ent, conf)
            phi_total: [B, T, 1] saude global
        """
        phi_dim = self._phi_dim(coords)
        phi_disp = self._phi_disp(coords)
        phi_ent = self._phi_ent(coords)
        phi_conf = self._phi_conf(confidence)

        # Stack componentes
        phi_components = torch.cat(
            [phi_dim, phi_disp, phi_ent, phi_conf], dim=-1
        )  # [B, T, 4]

        # phi_total = weighted sum
        phi_total = (phi_components * self.weights.unsqueeze(0).unsqueeze(0)).sum(
            dim=-1, keepdim=True
        )  # [B, T, 1]

        return phi_components, phi_total
