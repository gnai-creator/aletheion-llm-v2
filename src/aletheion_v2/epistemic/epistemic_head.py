"""
Epistemic Head: Orquestra DRM + MAD + VI -> EpistemicTomography.

Recebe hidden_states e attention_patterns do transformer
e produz tomografia epistemica completa por token.

Pipeline:
    hidden_states -> Q1Gate, Q2Gate -> q1, q2
    hidden_states -> ManifoldEmbedding -> coords [B,T,5]
    attn_patterns -> DirectionalField -> directions, dim_D
    coords + G -> GeodesicDistance -> distance
    coords + truth -> MADConfidence -> confidence
    coords + confidence -> PhiField -> phi_components, phi_total
    phi + coords -> IntentionalityVector -> vi_direction, vi_severity
    q1, q2 -> AdaptiveTemperature -> temperature
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
    """Cabeca epistemica: orquestra todos os modulos DRM/MAD/VI.

    Recebe hidden_states [B, T, d_model] e attention_patterns
    [B, n_layers, n_heads, T, T] e produz EpistemicTomography.

    Overhead de parametros: ~2.2M (~1.8% do modelo total).
    """

    def __init__(self, config: AletheionV2Config):
        super().__init__()
        self.config = config

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_patterns: torch.Tensor,
    ) -> EpistemicTomography:
        """Computa tomografia epistemica completa.

        Args:
            hidden_states: [B, T, d_model]
            attention_patterns: [B, n_layers, n_heads, T, T]

        Returns:
            EpistemicTomography com todos os campos preenchidos
        """
        # --- Gates Q1/Q2 ---
        q1 = self.q1_gate(hidden_states)  # [B, T, 1]
        q2 = self.q2_gate(hidden_states)  # [B, T, 1]

        # --- Temperatura adaptativa ---
        temperature = self.adaptive_temp(q1, q2)  # [B, T, 1]

        # --- DRM: Coordenadas 5D ---
        coords, anchor_dists = self.manifold_emb(hidden_states)  # [B,T,5], [B,T,A]

        # --- DRM: Metric tensor ---
        G = self.metric_tensor()  # [5, 5] SPD

        # --- DRM: Directional field ---
        directions, dim_D = self.directional_field(attention_patterns)

        # --- DRM: Geodesic distance ---
        truth_centroid = self.manifold_emb.anchors.truth_centroid  # [5]
        metric_distance = self.geodesic_dist(coords, truth_centroid, G)  # [B,T,1]

        # --- MAD: Confidence ---
        confidence, d_sq, tau_sq = self.mad_confidence(coords, truth_centroid)

        # --- VI: Phi field ---
        phi_components, phi_total = self.phi_field(coords, confidence)

        # --- VI: Direction + Severity ---
        vi_direction, vi_severity = self.vi(phi_components, phi_total, coords)

        # --- VI: Corrige confianca ---
        confidence = self.vi.correct_confidence(confidence, vi_severity)

        return EpistemicTomography(
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
        )

    def get_metric_tensor(self) -> torch.Tensor:
        """Retorna G para uso na loss de regularizacao."""
        return self.metric_tensor()

    def get_tau_sq(self) -> torch.Tensor:
        """Retorna tau^2 para uso na loss MAD."""
        return self.mad_confidence.tau.get_tau_sq()
