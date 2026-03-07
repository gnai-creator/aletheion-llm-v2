"""
Composite Loss: CE + VARO + VI + MAD + metric_reg + 8 losses adicionais + STP.

L_total = lambda_ce * CE
        + anneal * (lambda_varo * VARO
                  + lambda_vi * VI_reg
                  + lambda_mad * MAD_cal
                  + lambda_metric * metric_reg
                  + lambda_eidos * eidos_reg
                  + lambda_conflict * conflict_reg
                  + lambda_consciousness * consciousness_reg
                  + lambda_grounding * grounding_reg
                  + lambda_plasticity * plasticity_reg
                  + lambda_frontier * frontier_reg
                  + lambda_mopsi * mopsi_reg
                  + lambda_contrastive * contrastive_reg
                  + lambda_stp * stp_reg)

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
from aletheion_v2.loss.stp_loss import stp_loss


class AletheionV2Loss(nn.Module):
    """Loss composta do AletheionV2.

    Combina 14 componentes com pesos configuraveis e annealing.

    Args:
        config: AletheionV2Config
    """

    def __init__(self, config: AletheionV2Config):
        super().__init__()
        self.config = config

        # --- Core losses ---
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.varo_loss = VAROLoss()
        self.vi_reg = VIRegularization(config.vi_phi_critical)
        self.mad_cal = MADCalibrationLoss()

        # --- Extension losses (instanciadas sob demanda) ---
        self._init_extension_losses(config)

        # Pesos core
        self.lambda_ce = config.lambda_ce
        self.lambda_varo = config.lambda_varo
        self.lambda_vi = config.lambda_vi
        self.lambda_mad = config.lambda_mad
        self.lambda_metric = config.lambda_metric_reg

        # Pesos extensoes
        self.lambda_eidos = config.lambda_eidos
        self.lambda_conflict = config.lambda_conflict
        self.lambda_consciousness = config.lambda_consciousness
        self.lambda_grounding = config.lambda_grounding
        self.lambda_plasticity = config.lambda_plasticity
        self.lambda_frontier = config.lambda_frontier
        self.lambda_mopsi = config.lambda_mopsi
        self.lambda_contrastive = config.lambda_contrastive
        self.lambda_stp = config.lambda_stp
        self.enable_stp = config.enable_stp
        self.stp_num_triplets = config.stp_num_triplets

        # Annealing
        self.warmup_fraction = config.loss_warmup_fraction
        self.ramp_fraction = config.loss_ramp_fraction

    def _init_extension_losses(self, config: AletheionV2Config) -> None:
        """Instancia losses das extensoes habilitadas."""
        if config.enable_eidos:
            from aletheion_v2.loss.eidos_loss import EidosRegularization
            self.eidos_reg = EidosRegularization()

        if config.enable_filosofia3:
            from aletheion_v2.loss.conflict_loss import ConflictRegularization
            self.conflict_reg = ConflictRegularization()

        if config.enable_consciousness:
            from aletheion_v2.loss.consciousness_loss import (
                ConsciousnessRegularization,
            )
            self.consciousness_reg = ConsciousnessRegularization()

        if config.enable_grounding:
            from aletheion_v2.loss.grounding_loss import (
                GroundingRegularization,
            )
            self.grounding_reg = GroundingRegularization()

        if config.enable_plasticity:
            from aletheion_v2.loss.plasticity_loss import (
                PlasticityRegularization,
            )
            self.plasticity_reg = PlasticityRegularization()

        if config.enable_mpl:
            from aletheion_v2.loss.frontier_loss import (
                FrontierRegularization,
            )
            self.frontier_reg = FrontierRegularization()

        if config.enable_mopsi:
            from aletheion_v2.loss.mopsi_loss import MOPsiRegularization
            self.mopsi_reg = MOPsiRegularization()

        if config.enable_metacognitive:
            from aletheion_v2.loss.contrastive_loss import (
                ContrastiveRegularization,
            )
            self.contrastive_reg = ContrastiveRegularization()

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

    def _compute_extension_losses(
        self,
        tomography: EpistemicTomography,
        mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Computa todas as losses das extensoes.

        Args:
            tomography: tomografia epistemica
            mask: mascara de padding
            device: device dos tensores

        Returns:
            Dict com losses nomeadas
        """
        losses = {}
        zero = torch.tensor(0.0, device=device)

        # Eidos
        if hasattr(self, "eidos_reg") and tomography.axis_balance is not None:
            losses["eidos"] = self.eidos_reg(tomography.axis_balance, mask)
        else:
            losses["eidos"] = zero

        # Conflict
        if hasattr(self, "conflict_reg") and tomography.conflict_intensity is not None:
            losses["conflict"] = self.conflict_reg(
                tomography.conflict_intensity, mask,
            )
        else:
            losses["conflict"] = zero

        # Consciousness
        if hasattr(self, "consciousness_reg") and tomography.energy is not None:
            losses["consciousness"] = self.consciousness_reg(
                tomography.energy, mask,
            )
        else:
            losses["consciousness"] = zero

        # Grounding
        if (
            hasattr(self, "grounding_reg")
            and tomography.task_probs is not None
            and tomography.ambiguity_level is not None
        ):
            losses["grounding"] = self.grounding_reg(
                tomography.task_probs,
                tomography.ambiguity_level,
                tomography.q1,
                mask,
            )
        else:
            losses["grounding"] = zero

        # Plasticity
        if (
            hasattr(self, "plasticity_reg")
            and tomography.plasticity_remaining is not None
        ):
            losses["plasticity"] = self.plasticity_reg(
                tomography.plasticity_remaining, mask,
            )
        else:
            losses["plasticity"] = zero

        # Frontier
        if (
            hasattr(self, "frontier_reg")
            and tomography.frontier_score is not None
        ):
            # Novidade: 1 - (1 - frontier_score) = frontier_score (proxy)
            novelty_proxy = torch.ones_like(tomography.frontier_score) * 0.5
            if tomography.drm_coords is not None:
                # Usa variancia das coords como proxy de novidade
                coord_var = tomography.drm_coords.var(dim=-1, keepdim=True)
                novelty_proxy = coord_var.clamp(0, 1)
            losses["frontier"] = self.frontier_reg(
                tomography.frontier_score, novelty_proxy, mask,
            )
        else:
            losses["frontier"] = zero

        # MOPsi
        if (
            hasattr(self, "mopsi_reg")
            and tomography.psi is not None
            and tomography.conflict_intensity is not None
        ):
            losses["mopsi"] = self.mopsi_reg(
                tomography.psi,
                tomography.confidence,
                tomography.conflict_intensity,
                mask,
            )
        else:
            losses["mopsi"] = zero

        # Contrastive
        if (
            hasattr(self, "contrastive_reg")
            and tomography.divergence is not None
        ):
            losses["contrastive"] = self.contrastive_reg(
                tomography.divergence, mask,
            )
        else:
            losses["contrastive"] = zero

        return losses

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tomography: Optional[EpistemicTomography] = None,
        G: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        step: int = 0,
        total_steps: int = 1,
        hidden_states: Optional[torch.Tensor] = None,
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
            hidden_states: [B, T, D] hidden states (para STP loss)

        Returns:
            Dict com 'total', 'ce', 'varo', 'vi', 'mad', 'metric',
            'eidos', 'conflict', 'consciousness', 'grounding',
            'plasticity', 'frontier', 'mopsi', 'contrastive', 'stp', 'annealing'
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

        # STP loss (active from step 0, no annealing — it's a geometric
        # regularizer on hidden states, not an epistemic loss)
        stp = torch.tensor(0.0, device=ce.device)
        if self.enable_stp and hidden_states is not None:
            stp = stp_loss(hidden_states, num_triplets=self.stp_num_triplets)
        losses["stp"] = stp

        # Se sem tomografia, retorna CE + STP
        if tomography is None:
            losses["total"] = self.lambda_ce * ce + self.lambda_stp * stp
            for key in [
                "varo", "vi", "mad", "metric",
                "eidos", "conflict", "consciousness", "grounding",
                "plasticity", "frontier", "mopsi", "contrastive",
            ]:
                losses[key] = torch.tensor(0.0, device=ce.device)
            return losses

        # --- Core losses ---

        # VARO
        varo = self.varo_loss(
            tomography.q1, tomography.q2, logits, labels, mask
        )
        losses["varo"] = varo

        # VI Regularization
        vi = self.vi_reg(tomography.phi_total, tomography.vi_severity, mask)
        losses["vi"] = vi

        # MAD Calibration
        mad = self.mad_cal(tomography.confidence, logits, labels, mask)
        losses["mad"] = mad

        # Metric Regularization
        metric_loss = torch.tensor(0.0, device=ce.device)
        if G is not None:
            metric_loss = self.metric_regularization(G)
        losses["metric"] = metric_loss

        # --- Extension losses ---
        ext_losses = self._compute_extension_losses(
            tomography, mask, ce.device,
        )
        losses.update(ext_losses)

        # --- Total com annealing ---
        total = self.lambda_ce * ce

        # Core epistemicas
        total = total + anneal * self.lambda_varo * varo
        total = total + anneal * self.lambda_vi * vi
        total = total + anneal * self.lambda_mad * mad
        total = total + anneal * self.lambda_metric * metric_loss

        # Extensoes (mesmo annealing)
        total = total + anneal * self.lambda_eidos * ext_losses["eidos"]
        total = total + anneal * self.lambda_conflict * ext_losses["conflict"]
        total = total + anneal * self.lambda_consciousness * ext_losses["consciousness"]
        total = total + anneal * self.lambda_grounding * ext_losses["grounding"]
        total = total + anneal * self.lambda_plasticity * ext_losses["plasticity"]
        total = total + anneal * self.lambda_frontier * ext_losses["frontier"]
        total = total + anneal * self.lambda_mopsi * ext_losses["mopsi"]
        total = total + anneal * self.lambda_contrastive * ext_losses["contrastive"]

        # STP (no annealing — active from step 0)
        total = total + self.lambda_stp * stp

        losses["total"] = total

        return losses
