"""
Epistemic Head: Orquestra DRM + MAD + VI + 9 modulos adicionais -> EpistemicTomography.

Recebe hidden_states e attention_patterns do transformer
e produz tomografia epistemica completa por token.

Pipeline core:
    hidden_states -> Q1Gate, Q2Gate -> q1, q2
    hidden_states -> ManifoldEmbedding -> coords [B,T,5]
    attn_patterns -> DirectionalField -> directions, dim_D
    coords + G -> GeodesicDistance -> distance
    coords + truth -> MADConfidence -> confidence
    coords + confidence -> PhiField -> phi_components, phi_total
    phi + coords -> IntentionalityVector -> vi_direction, vi_severity
    q1, q2 -> AdaptiveTemperature -> temperature

Modulos adicionais (condicionais via config):
    Tier 1: EidosDecay, Filosofia3, SelfModel
    Tier 2: Grounding (Task+Ambiguity), PlasticityGate, FrontierHead
    Tier 3: MOPsi, CausalState, Metacognitive
"""

import torch
import torch.nn as nn
from typing import Optional

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.output import EpistemicTomography
from aletheion_v2.epistemic.gates import Q1Gate, Q2Gate, AdaptiveTemperature
from aletheion_v2.drm.manifold_embedding import ManifoldEmbedding
from aletheion_v2.drm.metric_tensor import LearnableMetricTensor
from aletheion_v2.drm.directional_field import DirectionalField
from aletheion_v2.drm.geodesic_distance import GeodesicDistance
from aletheion_v2.mad.confidence import MADConfidence
from aletheion_v2.vi.phi_field import PhiField
from aletheion_v2.vi.intentionality_vector import IntentionalityVector


class EpistemicHead(nn.Module):
    """Cabeca epistemica: orquestra todos os modulos DRM/MAD/VI + extensoes.

    Recebe hidden_states [B, T, d_model] e attention_patterns
    [B, n_layers, n_heads, T, T] e produz EpistemicTomography.

    Overhead de parametros:
    - Core: ~2.2M (~1.8% do modelo total)
    - Extensoes: ~364K (~0.30% adicional)
    """

    def __init__(self, config: AletheionV2Config):
        super().__init__()
        self.config = config

        # --- Core modules ---

        # Gates Q1/Q2
        self.q1_gate = Q1Gate(
            config.d_model, config.gate_hidden_dim,
            config.gate_num_layers, config.gate_dropout,
        )
        self.q2_gate = Q2Gate(
            config.d_model, config.gate_hidden_dim,
            config.gate_num_layers, config.gate_dropout,
        )

        # Temperatura adaptativa
        self.adaptive_temp = AdaptiveTemperature()

        # DRM: Manifold embedding
        self.manifold_emb = ManifoldEmbedding(
            config.d_model, config.drm_dim, config.drm_num_anchors,
        )

        # DRM: Metric tensor (G = L@L^T)
        self.metric_tensor = LearnableMetricTensor(
            config.drm_dim, config.metric_eps,
        )

        # DRM: Directional field (attn -> directions + dim_D)
        self.directional_field = DirectionalField(
            config.n_heads, config.n_layers, config.drm_dim,
        )

        # DRM: Geodesic distance
        self.geodesic_dist = GeodesicDistance(config.drm_dim)

        # MAD: Confidence
        self.mad_confidence = MADConfidence(
            config.drm_dim, config.mad_per_axis,
            config.mad_init_log_tau_sq,
        )

        # VI: Phi field
        self.phi_field = PhiField(
            config.drm_dim, config.vi_phi_weights,
            config.vi_ideal_conf_std,
        )

        # VI: Intentionality vector
        self.vi = IntentionalityVector(
            config.drm_dim, config.vi_phi_critical,
            config.vi_injection_strength, config.vi_confidence_penalty,
        )

        # --- Tier 1: Eidos, Filosofia3, Consciousness ---
        self._init_tier1(config)

        # --- Tier 2: Grounding, Plasticity, MPL ---
        self._init_tier2(config)

        # --- Tier 3: MOPsi, CausalState, Metacognitive ---
        self._init_tier3(config)

    def _init_tier1(self, config: AletheionV2Config) -> None:
        """Inicializa modulos Tier 1."""
        if config.enable_eidos:
            from aletheion_v2.eidos.eidos_decay import EidosDecay
            self.eidos = EidosDecay(
                drm_dim=config.drm_dim,
                base_decay=config.eidos_base_decay,
                base_reinforce=config.eidos_base_reinforce,
                dream_intensity=config.eidos_dream_intensity,
            )

        if config.enable_filosofia3:
            from aletheion_v2.filosofia3.conflict_head import PhiPsiConflictHead
            self.filosofia3 = PhiPsiConflictHead(
                quality_projection=config.filosofia3_quality_projection,
                analytical_weight=config.filosofia3_analytical_weight,
            )

        if config.enable_consciousness:
            from aletheion_v2.consciousness.self_model_head import SelfModelHead
            self.self_model = SelfModelHead(
                d_model=config.d_model,
                hidden_dim=config.consciousness_hidden_dim,
                energy_decay=config.consciousness_energy_decay,
            )

    def _init_tier2(self, config: AletheionV2Config) -> None:
        """Inicializa modulos Tier 2."""
        if config.enable_grounding:
            from aletheion_v2.grounding.task_head import TaskClassificationHead
            from aletheion_v2.grounding.ambiguity_head import AmbiguityHead
            self.task_head = TaskClassificationHead(
                d_model=config.d_model,
                hidden_dim=config.grounding_task_hidden_dim,
            )
            self.ambiguity_head = AmbiguityHead(
                d_model=config.d_model,
                hidden_dim=config.grounding_ambiguity_hidden_dim,
            )

        if config.enable_plasticity:
            from aletheion_v2.plasticity.plasticity_gate import PlasticityGate
            self.plasticity_gate = PlasticityGate(
                d_model=config.d_model,
                initial_budget=config.plasticity_initial_budget,
                depletion_rate=config.plasticity_depletion_rate,
            )

        if config.enable_mpl:
            from aletheion_v2.mpl.frontier_head import FrontierHead
            from aletheion_v2.mpl.density_tracker import DensityTracker
            self.frontier_head = FrontierHead(
                drm_dim=config.drm_dim,
                hidden_dim=config.mpl_hidden_dim,
            )
            # DensityTracker nao e nn.Module, armazenado como atributo
            self.density_tracker = DensityTracker(
                resolution=config.mpl_resolution,
                bandwidth=config.mpl_bandwidth,
                drm_dim=config.drm_dim,
            )

    def _init_tier3(self, config: AletheionV2Config) -> None:
        """Inicializa modulos Tier 3."""
        if config.enable_mopsi:
            from aletheion_v2.mopsi.human_state_head import (
                HumanStateHead, PhiPsiMediator,
            )
            self.human_state_head = HumanStateHead(
                d_model=config.d_model,
                hidden_dim=config.mopsi_hidden_dim,
            )
            self.phi_psi_mediator = PhiPsiMediator()

        if config.enable_causal_state:
            from aletheion_v2.causal_state.state_conditioning import (
                StateConditioning,
            )
            self.state_conditioning = StateConditioning(
                d_model=config.d_model,
                hidden_dim=config.causal_state_hidden_dim,
            )

        if config.enable_metacognitive:
            from aletheion_v2.metacognitive.contrastive_head import (
                ContrastiveHead,
            )
            proj_dim = config.metacognitive_proj_dim or None
            self.contrastive_head = ContrastiveHead(
                d_model=config.d_model,
                hidden_dim=config.metacognitive_hidden_dim,
                proj_dim=proj_dim,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_patterns: torch.Tensor,
        state_vector: Optional[torch.Tensor] = None,
        dream_mode: bool = False,
    ) -> EpistemicTomography:
        """Computa tomografia epistemica completa.

        Args:
            hidden_states: [B, T, d_model]
            attention_patterns: [B, n_layers, n_heads, T, T]
            state_vector: [B, 4] estado causal (opcional, para CausalState)
            dream_mode: se True, amplifica eidos decay

        Returns:
            EpistemicTomography com todos os campos preenchidos
        """
        # Opcional: CausalState condiciona hidden_states antes de tudo
        state_gate = None
        if hasattr(self, "state_conditioning") and state_vector is not None:
            sc_out = self.state_conditioning(hidden_states, state_vector)
            hidden_states = sc_out.conditioned_hidden
            state_gate = sc_out.gate_value

        # === CORE PIPELINE ===

        # Gates Q1/Q2
        q1 = self.q1_gate(hidden_states)
        q2 = self.q2_gate(hidden_states)

        # Temperatura adaptativa
        temperature = self.adaptive_temp(q1, q2)

        # DRM: Coordenadas 5D
        coords, anchor_dists = self.manifold_emb(hidden_states)

        # DRM: Metric tensor
        G = self.metric_tensor()

        # DRM: Directional field
        directions, dim_D = self.directional_field(attention_patterns)

        # DRM: Geodesic distance
        truth_centroid = self.manifold_emb.anchors.truth_centroid
        metric_distance = self.geodesic_dist(coords, truth_centroid, G)

        # MAD: Confidence (usa tensor metrico G para geometria real)
        confidence, d_sq, tau_sq = self.mad_confidence(coords, truth_centroid, G)

        # VI: Phi field
        phi_components, phi_total = self.phi_field(coords, confidence)

        # VI: Direction + Severity
        vi_direction, vi_severity = self.vi(phi_components, phi_total, coords)

        # VI: Corrige confianca
        confidence = self.vi.correct_confidence(confidence, vi_severity)

        # === TIER 1 ===

        eidos_weights = None
        axis_balance = None
        if hasattr(self, "eidos"):
            eidos_out = self.eidos(coords, confidence, dream_mode=dream_mode)
            eidos_weights = eidos_out.eidos_weights
            axis_balance = eidos_out.axis_balance

        conflict_intensity = None
        mode_probs = None
        if hasattr(self, "filosofia3"):
            conflict_out = self.filosofia3(phi_components, confidence)
            conflict_intensity = conflict_out.conflict_intensity
            mode_probs = conflict_out.mode_probs

        mood = None
        curiosity_val = None
        energy = None
        drives = None
        if hasattr(self, "self_model"):
            consciousness_out = self.self_model(
                hidden_states, q2, phi_total, confidence,
            )
            mood = consciousness_out.mood
            curiosity_val = consciousness_out.curiosity
            energy = consciousness_out.energy
            drives = consciousness_out.drives

        # === TIER 2 ===

        task_probs = None
        task_confidence = None
        ambiguity_level = None
        ambiguity_type = None
        if hasattr(self, "task_head"):
            task_out = self.task_head(hidden_states)
            task_probs = task_out.task_probs
            task_confidence = task_out.task_confidence
            amb_out = self.ambiguity_head(hidden_states)
            ambiguity_level = amb_out.ambiguity_level
            ambiguity_type = amb_out.ambiguity_type

        plasticity_remaining = None
        gate_val = None
        if hasattr(self, "plasticity_gate"):
            plast_out = self.plasticity_gate(hidden_states, vi_severity)
            plasticity_remaining = plast_out.plasticity_remaining
            gate_val = plast_out.gate_value

        frontier_score = None
        if hasattr(self, "frontier_head"):
            density_info = self.density_tracker.query(coords)
            frontier_out = self.frontier_head(coords, density_info.density)
            frontier_score = frontier_out.frontier_score
            # Atualiza tracker (no_grad internamente)
            if self.training:
                self.density_tracker.update(coords)

        # === TIER 3 ===

        human_state = None
        psi = None
        mediated = None
        if hasattr(self, "human_state_head"):
            hs_out = self.human_state_head(
                hidden_states, phi_components, confidence,
            )
            human_state = hs_out.human_state
            psi = hs_out.psi
            # Mediacao phi-psi (requer conflict do Filosofia3)
            if hasattr(self, "phi_psi_mediator") and conflict_intensity is not None:
                med_out = self.phi_psi_mediator(
                    phi_total, psi, conflict_intensity,
                )
                mediated = med_out.mediated_score

        divergence = None
        if hasattr(self, "contrastive_head"):
            contrast_out = self.contrastive_head(hidden_states)
            divergence = contrast_out.divergence

        return EpistemicTomography(
            # Core
            q1=q1,
            q2=q2,
            confidence=confidence,
            drm_coords=coords,
            directional_dim=dim_D,
            metric_distance=metric_distance,
            phi_components=phi_components,
            phi_total=phi_total,
            vi_direction=vi_direction,
            vi_severity=vi_severity,
            temperature=temperature,
            # Tier 1
            eidos_weights=eidos_weights,
            axis_balance=axis_balance,
            conflict_intensity=conflict_intensity,
            mode_probs=mode_probs,
            mood=mood,
            curiosity=curiosity_val,
            energy=energy,
            drives=drives,
            # Tier 2
            task_probs=task_probs,
            task_confidence=task_confidence,
            ambiguity_level=ambiguity_level,
            ambiguity_type=ambiguity_type,
            plasticity_remaining=plasticity_remaining,
            gate_value=gate_val,
            frontier_score=frontier_score,
            # Tier 3
            human_state=human_state,
            psi=psi,
            mediated_score=mediated,
            state_gate=state_gate,
            divergence=divergence,
        )

    def get_metric_tensor(self) -> torch.Tensor:
        """Retorna G para uso na loss de regularizacao."""
        return self.metric_tensor()

    def get_tau_sq(self) -> torch.Tensor:
        """Retorna tau^2 para uso na loss MAD."""
        return self.mad_confidence.tau.get_tau_sq()
