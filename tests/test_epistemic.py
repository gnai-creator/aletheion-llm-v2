"""
Testes do modulo Epistemic: gates Q1/Q2, temperatura, EpistemicHead.
"""

import torch
import pytest

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.epistemic.gates import (
    GateNetwork, Q1Gate, Q2Gate, AdaptiveTemperature,
)
from aletheion_v2.epistemic.epistemic_head import EpistemicHead


class TestGateNetwork:
    """Testes da rede gate generica."""

    def test_sigmoid_output_range(self):
        gate = GateNetwork(input_dim=32, output_activation="sigmoid")
        x = torch.randn(4, 32)
        out = gate(x)
        assert out.shape == (4, 1)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_softplus_output_range(self):
        gate = GateNetwork(input_dim=32, output_activation="softplus")
        x = torch.randn(4, 32)
        out = gate(x)
        assert (out >= 0).all()

    def test_3d_input(self):
        gate = GateNetwork(input_dim=32)
        x = torch.randn(2, 10, 32)
        out = gate(x)
        assert out.shape == (2, 10, 1)

    def test_gradient_flow(self):
        gate = GateNetwork(input_dim=32)
        x = torch.randn(4, 32, requires_grad=True)
        out = gate(x)
        out.sum().backward()
        assert x.grad is not None


class TestQ1Q2Gates:
    """Testes dos gates Q1 e Q2."""

    def test_q1_output(self):
        q1 = Q1Gate(d_model=64)
        hidden = torch.randn(2, 10, 64)
        out = q1(hidden)
        assert out.shape == (2, 10, 1)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_q2_output(self):
        q2 = Q2Gate(d_model=64)
        hidden = torch.randn(2, 10, 64)
        out = q2(hidden)
        assert out.shape == (2, 10, 1)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_q1_q2_independent(self):
        """Q1 e Q2 devem produzir saidas diferentes."""
        q1 = Q1Gate(d_model=64)
        q2 = Q2Gate(d_model=64)
        hidden = torch.randn(2, 10, 64)
        out1 = q1(hidden)
        out2 = q2(hidden)
        # Improvavel serem identicas com pesos aleatorios
        assert not torch.allclose(out1, out2)


class TestAdaptiveTemperature:
    """Testes da temperatura adaptativa."""

    def test_high_confidence_base_temp(self):
        temp = AdaptiveTemperature(base_temperature=1.0, tau_threshold=0.5)
        q1 = torch.full((2, 5, 1), 0.9)
        q2 = torch.full((2, 5, 1), 0.9)
        tau = temp(q1, q2)
        # c = 0.81 > threshold -> tau = 1.0
        assert torch.allclose(tau, torch.ones_like(tau))

    def test_low_confidence_high_temp(self):
        temp = AdaptiveTemperature(base_temperature=1.0, tau_threshold=0.5)
        q1 = torch.full((2, 5, 1), 0.3)
        q2 = torch.full((2, 5, 1), 0.3)
        tau = temp(q1, q2)
        # c = 0.09 < threshold -> tau = 1/0.09 ~ 11.1
        assert (tau > 1.0).all()

    def test_output_shape(self):
        temp = AdaptiveTemperature()
        q1 = torch.rand(3, 8, 1)
        q2 = torch.rand(3, 8, 1)
        tau = temp(q1, q2)
        assert tau.shape == (3, 8, 1)


class TestEpistemicHead:
    """Testes do EpistemicHead completo."""

    def setup_method(self):
        self.config = AletheionV2Config.small()
        self.head = EpistemicHead(self.config)
        self.B = 2
        self.T = 8

    def test_output_type(self):
        hidden = torch.randn(self.B, self.T, self.config.d_model)
        attn = torch.rand(self.B, self.config.n_layers, self.config.n_heads, self.T, self.T)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        tomo = self.head(hidden, attn)
        from aletheion_v2.core.output import EpistemicTomography
        assert isinstance(tomo, EpistemicTomography)

    def test_all_fields_shapes(self):
        hidden = torch.randn(self.B, self.T, self.config.d_model)
        attn = torch.rand(self.B, self.config.n_layers, self.config.n_heads, self.T, self.T)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        tomo = self.head(hidden, attn)

        assert tomo.q1.shape == (self.B, self.T, 1)
        assert tomo.q2.shape == (self.B, self.T, 1)
        assert tomo.confidence.shape == (self.B, self.T, 1)
        assert tomo.drm_coords.shape == (self.B, self.T, 5)
        assert tomo.directional_dim.shape == (self.B, self.T, 1)
        assert tomo.metric_distance.shape == (self.B, self.T, 1)
        assert tomo.phi_components.shape == (self.B, self.T, 4)
        assert tomo.phi_total.shape == (self.B, self.T, 1)
        assert tomo.vi_direction.shape == (self.B, self.T, 5)
        assert tomo.vi_severity.shape == (self.B, self.T, 1)
        assert tomo.temperature.shape == (self.B, self.T, 1)

    def test_confidence_range(self):
        hidden = torch.randn(self.B, self.T, self.config.d_model)
        attn = torch.rand(self.B, self.config.n_layers, self.config.n_heads, self.T, self.T)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        tomo = self.head(hidden, attn)

        assert (tomo.confidence >= 0).all()
        assert (tomo.confidence <= 1).all()

    def test_gradient_flow_through_head(self):
        hidden = torch.randn(self.B, self.T, self.config.d_model, requires_grad=True)
        attn = torch.rand(self.B, self.config.n_layers, self.config.n_heads, self.T, self.T)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        attn.requires_grad_(True)

        tomo = self.head(hidden, attn)
        loss = tomo.confidence.sum() + tomo.phi_total.sum()
        loss.backward()

        assert hidden.grad is not None

    def test_to_dict(self):
        hidden = torch.randn(self.B, self.T, self.config.d_model)
        attn = torch.rand(self.B, self.config.n_layers, self.config.n_heads, self.T, self.T)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        tomo = self.head(hidden, attn)

        d = tomo.to_dict()
        # Core 11 + extensoes habilitadas por default
        assert len(d) >= 11
        assert all(isinstance(v, torch.Tensor) for v in d.values())

    def test_get_metric_tensor(self):
        G = self.head.get_metric_tensor()
        assert G.shape == (5, 5)
        eigenvalues = torch.linalg.eigvalsh(G)
        assert (eigenvalues > 0).all()
