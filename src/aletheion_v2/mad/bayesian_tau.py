"""
Bayesian Tau: tau^2 aprendivel por eixo para MAD confidence.

Implementa tau^2 como nn.Parameter com positividade via exp(log_tau_sq).
Suporta tau por eixo (anisotropico) ou escalar (isotropico).

Formulas:
    tau_sq = exp(log_tau_sq)   -- garante positividade
    Sigma = diag(tau_sq)       -- covariancia diagonal
    d^2 = sum((p - mu)^2 / tau_sq)  -- Mahalanobis com Sigma diagonal

Posterior bayesiano (soft):
    A loss MAD_cal atualiza log_tau_sq via backprop,
    emulando update conjugado IG(alpha, beta) de forma diferenciavel.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class BayesianTau(nn.Module):
    """Tau^2 aprendivel para confianca MAD.

    Parametriza tau^2 em log-space para garantir positividade.
    Suporta modo per-axis (anisotropico) e escalar (isotropico).

    O tau controla a "largura" do kernel Gaussiano:
    - tau grande = tolerante (confianca cai devagar com distancia)
    - tau pequeno = estrito (confianca cai rapido)

    Args:
        drm_dim: Dimensao do manifold (5)
        per_axis: Se True, um tau por eixo; se False, tau unico
        init_log_tau_sq: Valor inicial de log(tau^2)
        min_tau_sq: Minimo de tau^2 (estabilidade)
        max_tau_sq: Maximo de tau^2 (evita degeneracao)
    """

    def __init__(
        self,
        drm_dim: int = 5,
        per_axis: bool = True,
        init_log_tau_sq: float = 0.0,
        min_tau_sq: float = 0.01,
        max_tau_sq: float = 10.0,
    ):
        super().__init__()
        self.drm_dim = drm_dim
        self.per_axis = per_axis
        self.min_tau_sq = min_tau_sq
        self.max_tau_sq = max_tau_sq

        if per_axis:
            # Um tau^2 por eixo: [drm_dim]
            self.log_tau_sq = nn.Parameter(
                torch.full((drm_dim,), init_log_tau_sq)
            )
        else:
            # Tau^2 escalar unico
            self.log_tau_sq = nn.Parameter(
                torch.tensor(init_log_tau_sq)
            )

    def get_tau_sq(self) -> torch.Tensor:
        """Retorna tau^2 com clamp para estabilidade.

        Returns:
            tau_sq: [drm_dim] ou escalar
        """
        tau_sq = torch.exp(self.log_tau_sq)
        return tau_sq.clamp(min=self.min_tau_sq, max=self.max_tau_sq)

    def get_covariance_diag(self) -> torch.Tensor:
        """Retorna diagonal da covariancia (= tau_sq por eixo).

        Returns:
            diag: [drm_dim]
        """
        tau_sq = self.get_tau_sq()
        if not self.per_axis:
            tau_sq = tau_sq.expand(self.drm_dim)
        return tau_sq

    def mahalanobis_sq(
        self,
        coords: torch.Tensor,
        centroid: torch.Tensor,
    ) -> torch.Tensor:
        """Distancia de Mahalanobis^2 com Sigma diagonal.

        d^2 = sum_i (p_i - mu_i)^2 / tau_sq_i

        Args:
            coords: [B, T, drm_dim]
            centroid: [drm_dim]

        Returns:
            d_sq: [B, T, 1]
        """
        tau_sq = self.get_tau_sq()  # [drm_dim] ou escalar
        delta = coords - centroid.unsqueeze(0).unsqueeze(0)  # [B, T, D]

        if self.per_axis:
            # Anisotropico: divide por tau_sq por eixo
            d_sq = (delta ** 2 / tau_sq.unsqueeze(0).unsqueeze(0)).sum(
                dim=-1, keepdim=True
            )
        else:
            # Isotropico: divide por tau_sq escalar
            d_sq = (delta ** 2).sum(dim=-1, keepdim=True) / tau_sq

        return d_sq

    def forward(
        self,
        coords: torch.Tensor,
        centroid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computa Mahalanobis distance e tau_sq.

        Args:
            coords: [B, T, drm_dim]
            centroid: [drm_dim]

        Returns:
            d_sq: [B, T, 1] distancia ao quadrado
            tau_sq: [drm_dim] ou escalar (para loss)
        """
        d_sq = self.mahalanobis_sq(coords, centroid)
        return d_sq, self.get_tau_sq()

    def calibration_target(
        self,
        coords: torch.Tensor,
        centroid: torch.Tensor,
        correct_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computa target de calibracao para loss MAD.

        Idea: tokens corretos devem ter d^2 baixo,
        tokens incorretos devem ter d^2 alto.

        Args:
            coords: [B, T, drm_dim]
            centroid: [drm_dim]
            correct_mask: [B, T] bool (True = token correto)

        Returns:
            target_d_sq: [B, T, 1] distancia target
        """
        d_sq = self.mahalanobis_sq(coords, centroid)

        # Target: corretos -> d_sq proximo de 0, incorretos -> d_sq alto
        target = torch.where(
            correct_mask.unsqueeze(-1),
            torch.zeros_like(d_sq),  # Corretos: distancia 0
            torch.ones_like(d_sq) * 2.0,  # Incorretos: distancia 2
        )
        return target
