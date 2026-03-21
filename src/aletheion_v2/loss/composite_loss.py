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
        self.lambda_metric_diversity = getattr(
            config, "lambda_metric_diversity", 0.05
        )
        self.lambda_stp = config.lambda_stp
        self.enable_stp = config.enable_stp
        self.stp_num_triplets = config.stp_num_triplets

        # Annealing
        self.warmup_fraction = config.loss_warmup_fraction
        self.ramp_fraction = config.loss_ramp_fraction

        # Lambda decay
        self.lambda_decay_mode = getattr(config, "lambda_decay_mode", "none")
        self.lambda_decay_k = getattr(config, "lambda_decay_k", 0.0003)
        # Store initial lambdas for exponential decay (λ(t) = λ_0 × e^(-k*t))
        self._lambda_init = {
            "stp": self.lambda_stp,
            "varo": self.lambda_varo,
            "vi": self.lambda_vi,
            "mad": self.lambda_mad,
            "metric": self.lambda_metric,
            "eidos": self.lambda_eidos,
            "conflict": self.lambda_conflict,
            "consciousness": self.lambda_consciousness,
            "grounding": self.lambda_grounding,
            "plasticity": self.lambda_plasticity,
            "frontier": self.lambda_frontier,
            "mopsi": self.lambda_mopsi,
            "contrastive": self.lambda_contrastive,
            "metric_diversity": self.lambda_metric_diversity,
        }
        self._decay_base_step = 0  # set on first forward (for resume)

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

    def _apply_lambda_decay(self, step: int) -> None:
        """Exponential decay: λ(t) = λ_0 × e^(-k × t_relative).

        t_relative is steps since training started (or resumed).
        """
        if self.lambda_decay_mode != "exponential":
            return
        import math
        # Initialize base step on first call (handles resume)
        if self._decay_base_step == 0 and step > 0:
            self._decay_base_step = step
            return
        t = step - self._decay_base_step
        if t <= 0:
            return
        factor = math.exp(-self.lambda_decay_k * t)
        self.lambda_stp = self._lambda_init["stp"] * factor
        self.lambda_varo = self._lambda_init["varo"] * factor
        self.lambda_vi = self._lambda_init["vi"] * factor
        self.lambda_mad = self._lambda_init["mad"] * factor
        self.lambda_metric = self._lambda_init["metric"] * factor
        self.lambda_eidos = self._lambda_init["eidos"] * factor
        self.lambda_conflict = self._lambda_init["conflict"] * factor
        self.lambda_consciousness = self._lambda_init["consciousness"] * factor
        self.lambda_grounding = self._lambda_init["grounding"] * factor
        self.lambda_plasticity = self._lambda_init["plasticity"] * factor
        self.lambda_frontier = self._lambda_init["frontier"] * factor
        self.lambda_mopsi = self._lambda_init["mopsi"] * factor
        self.lambda_contrastive = self._lambda_init["contrastive"] * factor
        self.lambda_metric_diversity = self._lambda_init["metric_diversity"] * factor

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

    def metric_regularization(
        self,
        G: torch.Tensor,
        metric_net: Optional[object] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Regularizacao do tensor metrico G (constante ou variavel).

        Para G constante [D,D]: condition proxy + scale penalty.
        Para G(x) [B,T,D,D]: condition proxy por ponto + suavidade
        do campo via perturbacao.

        Args:
            G: [D, D] ou [B, T, D, D] tensor metrico
            metric_net: MetricNet para smoothness (opcional)
            coords: [B, T, D] coordenadas (para smoothness)

        Returns:
            loss: escalar
        """
        # Guard against NaN/Inf in G
        if torch.isnan(G).any() or torch.isinf(G).any():
            return torch.tensor(0.0, device=G.device, dtype=G.dtype)

        if G.dim() == 2:
            return self._metric_reg_constant(G)
        return self._metric_reg_field(G, metric_net, coords)

    def _metric_reg_constant(self, G: torch.Tensor) -> torch.Tensor:
        """Regularizacao para G constante [D, D]."""
        diag = G.diagonal()
        frob_sq = G.pow(2).sum()
        trace = diag.sum()
        dim = G.shape[0]

        normalized_frob = frob_sq / (trace.pow(2).clamp(min=1e-8))
        condition_proxy = (normalized_frob - 1.0 / dim).clamp(min=0.0)
        scale_penalty = (trace / dim - 1.0).pow(2)

        return condition_proxy + 0.1 * scale_penalty

    def _metric_reg_field(
        self,
        G: torch.Tensor,
        metric_net: Optional[object] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Regularizacao para G(x) variavel [B, T, D, D].

        Penaliza:
        1. Condition number alto por ponto (Frobenius proxy)
        2. Escala excessiva por ponto
        3. Variacao abrupta de G (suavidade do campo)
        """
        dim = G.shape[-1]
        diag = G.diagonal(dim1=-2, dim2=-1)  # [B, T, D]
        frob_sq = G.pow(2).sum(dim=(-2, -1))  # [B, T]
        trace = diag.sum(dim=-1)  # [B, T]

        # Condition proxy por ponto (media sobre batch)
        normalized_frob = frob_sq / trace.pow(2).clamp(min=1e-8)
        condition_proxy = (normalized_frob - 1.0 / dim).clamp(min=0.0).mean()

        # Escala por ponto
        scale_penalty = (trace / dim - 1.0).pow(2).mean()

        loss = condition_proxy + 0.1 * scale_penalty

        # Suavidade: G nao deve variar abruptamente
        lambda_smooth = getattr(self.config, "lambda_metric_smoothness", 0.1)
        if metric_net is not None and coords is not None:
            eps = 0.01
            noise = torch.randn_like(coords) * eps
            coords_perturbed = (coords + noise).clamp(0, 1)
            G_perturbed = metric_net(coords_perturbed)
            smoothness = (G - G_perturbed.detach()).pow(2).sum(
                dim=(-2, -1)
            ).mean()
            loss = loss + lambda_smooth * smoothness

        return loss

    def _metric_diversity_loss(self, G: torch.Tensor) -> torch.Tensor:
        """Penaliza G(x) constante entre posicoes.

        Para G constante [D, D]: retorna 0 (nao aplicavel).
        Para G(x) [B, T, D, D]: incentiva variancia entre posicoes
        via -log(var), bounded por clamp.

        Args:
            G: [D, D] ou [B, T, D, D] tensor metrico

        Returns:
            loss: escalar
        """
        if G.dim() == 2:
            return torch.tensor(0.0, device=G.device)

        # G: [B, T, D, D] -> flatten matrix dims
        B, T, D, _ = G.shape
        G_flat = G.reshape(B, T, D * D)

        # Variancia media ao longo de T para cada elemento da matriz
        var_per_elem = G_flat.var(dim=1)  # [B, D*D]
        mean_var = var_per_elem.mean()

        # -log(var + eps): gradiente forte quando var baixa, bounded
        loss = -torch.log(mean_var + 1e-4)
        loss = torch.clamp(loss, min=0.0)

        return loss

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
        **kwargs,
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

        # Lambda decay
        self._apply_lambda_decay(step)

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
                "varo", "vi", "mad", "metric", "metric_diversity",
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
            metric_loss = self.metric_regularization(
                G,
                metric_net=kwargs.get("metric_net"),
                coords=tomography.drm_coords if tomography is not None else None,
            )
        losses["metric"] = metric_loss

        # Metric diversity (penaliza G(x) constante entre posicoes)
        metric_div = torch.tensor(0.0, device=ce.device)
        if (
            self.lambda_metric_diversity > 0
            and tomography is not None
            and tomography.metric_G is not None
        ):
            metric_div = self._metric_diversity_loss(tomography.metric_G)
        losses["metric_diversity"] = metric_div

        # --- Extension losses ---
        ext_losses = self._compute_extension_losses(
            tomography, mask, ce.device,
        )
        losses.update(ext_losses)

        # NaN guard: replace any NaN individual losses with 0 (keep grad chain)
        _zero = torch.zeros(1, device=ce.device, dtype=ce.dtype, requires_grad=True).squeeze()
        def safe(val):
            if torch.isnan(val) or torch.isinf(val):
                return _zero
            return val

        # --- Total com annealing (all losses wrapped with safe()) ---
        total = self.lambda_ce * safe(ce)
        total = total + anneal * self.lambda_varo * safe(varo)
        total = total + anneal * self.lambda_vi * safe(vi)
        total = total + anneal * self.lambda_mad * safe(mad)
        total = total + anneal * self.lambda_metric * safe(metric_loss)
        total = total + anneal * self.lambda_metric_diversity * safe(metric_div)
        total = total + anneal * self.lambda_eidos * safe(ext_losses["eidos"])
        total = total + anneal * self.lambda_conflict * safe(ext_losses["conflict"])
        total = total + anneal * self.lambda_consciousness * safe(ext_losses["consciousness"])
        total = total + anneal * self.lambda_grounding * safe(ext_losses["grounding"])
        total = total + anneal * self.lambda_plasticity * safe(ext_losses["plasticity"])
        total = total + anneal * self.lambda_frontier * safe(ext_losses["frontier"])
        total = total + anneal * self.lambda_mopsi * safe(ext_losses["mopsi"])
        total = total + anneal * self.lambda_contrastive * safe(ext_losses["contrastive"])
        total = total + self.lambda_stp * safe(stp)

        losses["total"] = total

        return losses
