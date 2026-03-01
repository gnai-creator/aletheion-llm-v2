"""
Composite Loss: CE + VARO + VI + MAD + metric_reg.

L_total = lambda_ce * CE
        + lambda_varo * VARO
        + lambda_vi * VI_reg
        + lambda_mad * MAD_cal
        + lambda_metric * metric_reg

Annealing: primeiros warmup_fraction steps so CE,
depois ramp linear ate ramp_fraction do treino total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.output import EpistemicTomography
from aletheion_v2.loss.varo_loss import VAROLoss
from aletheion_v2.loss.vi_regularization import VIRegularization
from aletheion_v2.loss.mad_calibration import MADCalibrationLoss


class AletheionV2Loss(nn.Module):
    """Loss composta do AletheionV2.

    Combina 5 componentes com pesos configuraveis e annealing.

    Args:
        config: AletheionV2Config
    """

    def __init__(self, config: AletheionV2Config):
        super().__init__()
        self.config = config

        # Componentes
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.varo_loss = VAROLoss()
        self.vi_reg = VIRegularization(config.vi_phi_critical)
        self.mad_cal = MADCalibrationLoss()

        # Pesos
        self.lambda_ce = config.lambda_ce
        self.lambda_varo = config.lambda_varo
        self.lambda_vi = config.lambda_vi
        self.lambda_mad = config.lambda_mad
        self.lambda_metric = config.lambda_metric_reg

        # Annealing
        self.warmup_fraction = config.loss_warmup_fraction
        self.ramp_fraction = config.loss_ramp_fraction

    def _get_annealing_factor(
        self, step: int, total_steps: int
    ) -> float:
        """Computa fator de annealing para losses epistemicas.

        0.0 durante warmup, ramp linear ate 1.0 em ramp_fraction.

        Args:
            step: step atual
            total_steps: total de steps

        Returns:
            factor: float em [0, 1]
        """
        warmup_end = int(total_steps * self.warmup_fraction)
        ramp_end = int(total_steps * self.ramp_fraction)

        if step < warmup_end:
            return 0.0
        if step >= ramp_end:
            return 1.0

        # Ramp linear
        progress = (step - warmup_end) / max(ramp_end - warmup_end, 1)
        return progress

    def metric_regularization(self, G: torch.Tensor) -> torch.Tensor:
        """Regularizacao do condition number de G.

        Penaliza G muito anisotropico (eigenvalues muito diferentes).

        Args:
            G: [D, D] tensor metrico

        Returns:
            loss: escalar (log condition number)
        """
        eigenvalues = torch.linalg.eigvalsh(G)
        # Log condition number
        kappa = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
        return torch.log(kappa + 1.0)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tomography: Optional[EpistemicTomography] = None,
        G: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        step: int = 0,
        total_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Computa loss composta.

        Args:
            logits: [B, T, V]
            labels: [B, T]
            tomography: EpistemicTomography (pode ser None)
            G: [D, D] tensor metrico (para regularizacao)
            mask: [B, T] mascara (1 = valido, 0 = padding)
            step: step atual (para annealing)
            total_steps: total de steps

        Returns:
            Dict com 'total', 'ce', 'varo', 'vi', 'mad', 'metric', 'annealing'
        """
        B, T, V = logits.shape
        losses = {}

        # --- Cross-Entropy ---
        ce = self.ce_loss(logits.view(-1, V), labels.view(-1))
        ce = ce.view(B, T)
        if mask is not None:
            ce = (ce * mask).sum() / (mask.sum() + 1e-8)
        else:
            ce = ce.mean()
        losses["ce"] = ce

        # --- Annealing ---
        anneal = self._get_annealing_factor(step, total_steps)
        losses["annealing"] = torch.tensor(anneal)

        # Se sem tomografia, retorna so CE
        if tomography is None:
            losses["total"] = self.lambda_ce * ce
            losses["varo"] = torch.tensor(0.0, device=ce.device)
            losses["vi"] = torch.tensor(0.0, device=ce.device)
            losses["mad"] = torch.tensor(0.0, device=ce.device)
            losses["metric"] = torch.tensor(0.0, device=ce.device)
            return losses

        # --- VARO ---
        varo = self.varo_loss(
            tomography.q1, tomography.q2, logits, labels, mask
        )
        losses["varo"] = varo

        # --- VI Regularization ---
        vi = self.vi_reg(tomography.phi_total, tomography.vi_severity, mask)
        losses["vi"] = vi

        # --- MAD Calibration ---
        mad = self.mad_cal(tomography.confidence, logits, labels, mask)
        losses["mad"] = mad

        # --- Metric Regularization ---
        metric_loss = torch.tensor(0.0, device=ce.device)
        if G is not None:
            metric_loss = self.metric_regularization(G)
        losses["metric"] = metric_loss

        # --- Total com annealing ---
        total = self.lambda_ce * ce
        total = total + anneal * self.lambda_varo * varo
        total = total + anneal * self.lambda_vi * vi
        total = total + anneal * self.lambda_mad * mad
        total = total + anneal * self.lambda_metric * metric_loss

        losses["total"] = total

        return losses
