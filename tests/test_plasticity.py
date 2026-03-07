"""Testes do modulo PlasticityGate."""

import torch
import pytest

from aletheion_v2.plasticity.plasticity_gate import PlasticityGate, PlasticityOutput
from aletheion_v2.loss.plasticity_loss import PlasticityRegularization


class TestPlasticityGate:
    """Testes do PlasticityGate."""

    def setup_method(self):
        self.d_model = 256
        self.gate = PlasticityGate(d_model=self.d_model)
        self.B = 2
        self.T = 8

    def test_output_type(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        severity = torch.rand(self.B, self.T, 1)
        out = self.gate(hidden, severity)
        assert isinstance(out, PlasticityOutput)

    def test_output_shapes(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        severity = torch.rand(self.B, self.T, 1)
        out = self.gate(hidden, severity)
        assert out.plasticity_remaining.shape == (self.B, self.T, 1)
        assert out.gate_value.shape == (self.B, self.T, 1)
        assert out.cost_per_token.shape == (self.B, self.T, 1)

    def test_plasticity_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        severity = torch.rand(self.B, self.T, 1)
        out = self.gate(hidden, severity)
        assert (out.plasticity_remaining >= 0).all()
        assert (out.plasticity_remaining <= 1).all()

    def test_gate_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        severity = torch.rand(self.B, self.T, 1)
        out = self.gate(hidden, severity)
        assert (out.gate_value >= 0).all()
        assert (out.gate_value <= 1).all()

    def test_cost_positive(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        severity = torch.rand(self.B, self.T, 1)
        out = self.gate(hidden, severity)
        assert (out.cost_per_token >= 0).all()

    def test_plasticity_decreases(self):
        """Plasticidade deve tender a diminuir ao longo de T."""
        hidden = torch.randn(self.B, 32, self.d_model)
        severity = torch.rand(self.B, 32, 1)
        out = self.gate(hidden, severity)
        first_half = out.plasticity_remaining[:, :16, :].mean()
        second_half = out.plasticity_remaining[:, 16:, :].mean()
        # Budget decrescente + custo cumulativo
        assert first_half >= second_half

    def test_gradient_flow(self):
        hidden = torch.randn(self.B, self.T, self.d_model, requires_grad=True)
        severity = torch.rand(self.B, self.T, 1)
        out = self.gate(hidden, severity)
        loss = out.gate_value.sum()
        loss.backward()
        assert hidden.grad is not None

    def test_param_count(self):
        gate_768 = PlasticityGate(d_model=768)
        n_params = sum(p.numel() for p in gate_768.parameters())
        # cost_estimator: 768*16+16+16*1+1 = 12305
        # gate_proj: 2*8+8+8*1+1 = 33
        # ~12338
        assert 12000 <= n_params <= 13000


class TestPlasticityRegularization:
    """Testes da loss de plasticidade."""

    def setup_method(self):
        self.loss_fn = PlasticityRegularization(min_plasticity=0.3)

    def test_high_plasticity_zero_loss(self):
        plasticity = torch.ones(2, 8, 1) * 0.8
        loss = self.loss_fn(plasticity)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_low_plasticity_positive_loss(self):
        plasticity = torch.ones(2, 8, 1) * 0.1
        loss = self.loss_fn(plasticity)
        assert loss.item() > 0

    def test_gradient_flow(self):
        plasticity = (torch.rand(2, 8, 1) * 0.5).requires_grad_(True)
        plasticity.retain_grad()
        loss = self.loss_fn(plasticity)
        loss.backward()
        assert plasticity.grad is not None
