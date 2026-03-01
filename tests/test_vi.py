"""
Testes do modulo VI: PhiField e IntentionalityVector.
"""

import torch
import pytest

from aletheion_v2.vi.phi_field import PhiField
from aletheion_v2.vi.intentionality_vector import IntentionalityVector
from aletheion_v2.mad.bayesian_tau import BayesianTau
from aletheion_v2.mad.confidence import MADConfidence


class TestPhiField:
    """Testes do campo phi(M)."""

    def setup_method(self):
        self.phi = PhiField(drm_dim=5)

    def test_output_shapes(self):
        coords = torch.rand(2, 10, 5)
        confidence = torch.rand(2, 10, 1)
        components, total = self.phi(coords, confidence)
        assert components.shape == (2, 10, 4)
        assert total.shape == (2, 10, 1)

    def test_phi_total_range(self):
        coords = torch.rand(4, 8, 5)
        confidence = torch.rand(4, 8, 1)
        _, total = self.phi(coords, confidence)
        assert (total >= 0).all()
        assert (total <= 1).all()

    def test_uniform_coords_high_diversity(self):
        """Coordenadas uniformemente distribuidas devem ter alta diversidade."""
        # Coords espalhados uniformemente em [0, 1]
        coords = torch.linspace(0, 1, 50).unsqueeze(0).unsqueeze(-1)
        coords = coords.expand(1, 50, 5)
        confidence = torch.full((1, 50, 1), 0.5)
        components, _ = self.phi(coords, confidence)
        # phi_dim (diversidade) deve ser razoavel
        assert components[0, 0, 0].item() > 0.3

    def test_gradient_flow(self):
        coords = torch.rand(2, 10, 5, requires_grad=True)
        confidence = torch.rand(2, 10, 1, requires_grad=True)
        _, total = self.phi(coords, confidence)
        total.sum().backward()
        assert coords.grad is not None


class TestIntentionalityVector:
    """Testes do VI."""

    def setup_method(self):
        self.vi = IntentionalityVector(drm_dim=5, phi_critical=0.5)

    def test_output_shapes(self):
        phi_comp = torch.rand(2, 10, 4)
        phi_total = torch.rand(2, 10, 1)
        coords = torch.rand(2, 10, 5)
        direction, severity = self.vi(phi_comp, phi_total, coords)
        assert direction.shape == (2, 10, 5)
        assert severity.shape == (2, 10, 1)

    def test_severity_range(self):
        phi_comp = torch.rand(2, 10, 4)
        phi_total = torch.rand(2, 10, 1)
        coords = torch.rand(2, 10, 5)
        _, severity = self.vi(phi_comp, phi_total, coords)
        assert (severity >= 0).all()
        assert (severity <= 1).all()

    def test_low_phi_high_severity(self):
        """phi baixo deve resultar em severity alta (analitico)."""
        phi_comp = torch.full((1, 5, 4), 0.2)
        phi_total = torch.full((1, 5, 1), 0.1)  # Bem abaixo de 0.5
        coords = torch.rand(1, 5, 5)
        _, severity = self.vi(phi_comp, phi_total, coords)
        # Severity analitica = sqrt(1 - 0.1/0.5) = sqrt(0.8) ~ 0.89
        # Blend 70/30, severity deve ser alta
        assert severity.mean().item() > 0.4

    def test_high_phi_low_severity(self):
        """phi alto deve resultar em severity baixa."""
        phi_comp = torch.full((1, 5, 4), 0.9)
        phi_total = torch.full((1, 5, 1), 0.9)  # Bem acima de 0.5
        coords = torch.rand(1, 5, 5)
        _, severity = self.vi(phi_comp, phi_total, coords)
        # Severity analitica = sqrt(max(0, 1 - 0.9/0.5)) = 0
        # Blend com componente aprendido, deve ser baixa
        assert severity.mean().item() < 0.5

    def test_confidence_correction(self):
        """Correcao de confianca deve reduzir com severity."""
        confidence = torch.full((2, 5, 1), 0.8)
        severity = torch.full((2, 5, 1), 0.5)
        corrected = self.vi.correct_confidence(confidence, severity)
        # corrected = 0.8 * (1 - 0.5 * 0.4) = 0.8 * 0.8 = 0.64
        assert (corrected < confidence).all()
        assert (corrected >= 0).all()
        assert (corrected <= 1).all()

    def test_gradient_flow(self):
        phi_comp = torch.rand(2, 5, 4, requires_grad=True)
        phi_total = torch.rand(2, 5, 1, requires_grad=True)
        coords = torch.rand(2, 5, 5, requires_grad=True)
        direction, severity = self.vi(phi_comp, phi_total, coords)
        loss = direction.sum() + severity.sum()
        loss.backward()
        assert phi_comp.grad is not None
        assert coords.grad is not None


class TestBayesianTau:
    """Testes do BayesianTau."""

    def test_per_axis_shape(self):
        tau = BayesianTau(drm_dim=5, per_axis=True)
        tau_sq = tau.get_tau_sq()
        assert tau_sq.shape == (5,)

    def test_isotropic_shape(self):
        tau = BayesianTau(drm_dim=5, per_axis=False)
        tau_sq = tau.get_tau_sq()
        assert tau_sq.shape == ()

    def test_positivity(self):
        tau = BayesianTau(drm_dim=5)
        tau_sq = tau.get_tau_sq()
        assert (tau_sq > 0).all()

    def test_mahalanobis_zero_at_centroid(self):
        tau = BayesianTau(drm_dim=5)
        centroid = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        coords = centroid.unsqueeze(0).unsqueeze(0)  # [1, 1, 5]
        d_sq = tau.mahalanobis_sq(coords, centroid)
        assert d_sq.item() < 1e-6

    def test_gradient_flow(self):
        tau = BayesianTau(drm_dim=5)
        centroid = torch.tensor([0.5] * 5)
        coords = torch.rand(2, 5, 5)
        d_sq, tau_sq = tau(coords, centroid)
        d_sq.sum().backward()
        assert tau.log_tau_sq.grad is not None


class TestMADConfidence:
    """Testes da confianca MAD."""

    def test_confidence_at_truth(self):
        mad = MADConfidence(drm_dim=5)
        truth = torch.tensor([0.5] * 5)
        coords = truth.unsqueeze(0).unsqueeze(0)
        conf, d_sq, tau_sq = mad(coords, truth)
        # Distancia 0 -> confianca ~1
        assert conf.item() > 0.99

    def test_confidence_far_from_truth(self):
        mad = MADConfidence(drm_dim=5)
        truth = torch.tensor([0.1] * 5)
        far = torch.tensor([[[0.9, 0.9, 0.9, 0.9, 0.9]]])
        conf, _, _ = mad(far, truth)
        # Distancia grande -> confianca baixa
        assert conf.item() < 0.5

    def test_output_shapes(self):
        mad = MADConfidence(drm_dim=5)
        truth = torch.tensor([0.5] * 5)
        coords = torch.rand(2, 10, 5)
        conf, d_sq, tau_sq = mad(coords, truth)
        assert conf.shape == (2, 10, 1)
        assert d_sq.shape == (2, 10, 1)
