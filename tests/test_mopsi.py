"""Testes do modulo MOPsi: HumanStateHead e PhiPsiMediator."""

import torch
import pytest

from aletheion_v2.mopsi.human_state_head import (
    HumanStateHead, HumanStateOutput,
    PhiPsiMediator, MediationOutput,
)
from aletheion_v2.loss.mopsi_loss import MOPsiRegularization


class TestHumanStateHead:
    """Testes do HumanStateHead."""

    def setup_method(self):
        self.d_model = 256
        self.head = HumanStateHead(d_model=self.d_model, hidden_dim=32)
        self.B = 2
        self.T = 8

    def test_output_type(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        phi = torch.rand(self.B, self.T, 4)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, phi, conf)
        assert isinstance(out, HumanStateOutput)

    def test_output_shapes(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        phi = torch.rand(self.B, self.T, 4)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, phi, conf)
        assert out.human_state.shape == (self.B, self.T, 5)
        assert out.psi.shape == (self.B, self.T, 1)

    def test_state_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        phi = torch.rand(self.B, self.T, 4)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, phi, conf)
        assert (out.human_state >= 0).all()
        assert (out.human_state <= 1).all()

    def test_psi_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        phi = torch.rand(self.B, self.T, 4)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, phi, conf)
        assert (out.psi >= 0).all()
        assert (out.psi <= 1).all()

    def test_gradient_flow(self):
        hidden = torch.randn(self.B, self.T, self.d_model, requires_grad=True)
        phi = torch.rand(self.B, self.T, 4)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, phi, conf)
        loss = out.psi.sum() + out.human_state.sum()
        loss.backward()
        assert hidden.grad is not None

    def test_param_count(self):
        head_768 = HumanStateHead(d_model=768, hidden_dim=32)
        n_params = sum(p.numel() for p in head_768.parameters())
        # state_net: 768*32+32+32*5+5 = 24741
        # psi_net: 10*32+32+32*1+1 = 385
        # ~25126
        assert 24000 <= n_params <= 26000


class TestPhiPsiMediator:
    """Testes do PhiPsiMediator."""

    def setup_method(self):
        self.mediator = PhiPsiMediator()
        self.B = 2
        self.T = 8

    def test_output_type(self):
        phi = torch.rand(self.B, self.T, 1)
        psi = torch.rand(self.B, self.T, 1)
        conflict = torch.rand(self.B, self.T, 1)
        out = self.mediator(phi, psi, conflict)
        assert isinstance(out, MediationOutput)

    def test_output_shapes(self):
        phi = torch.rand(self.B, self.T, 1)
        psi = torch.rand(self.B, self.T, 1)
        conflict = torch.rand(self.B, self.T, 1)
        out = self.mediator(phi, psi, conflict)
        assert out.mediated_score.shape == (self.B, self.T, 1)
        assert out.phi_weight.shape == (self.B, self.T, 1)

    def test_mediated_range(self):
        phi = torch.rand(self.B, self.T, 1)
        psi = torch.rand(self.B, self.T, 1)
        conflict = torch.rand(self.B, self.T, 1)
        out = self.mediator(phi, psi, conflict)
        assert (out.mediated_score >= 0).all()
        assert (out.mediated_score <= 1).all()

    def test_high_conflict_phi_weight(self):
        """Alto conflito -> peso phi mais alto."""
        phi = torch.rand(self.B, self.T, 1)
        psi = torch.rand(self.B, self.T, 1)
        high_conflict = torch.ones(self.B, self.T, 1) * 0.9
        low_conflict = torch.ones(self.B, self.T, 1) * 0.1
        out_high = self.mediator(phi, psi, high_conflict)
        out_low = self.mediator(phi, psi, low_conflict)
        assert (out_high.phi_weight > out_low.phi_weight).all()

    def test_gradient_flow(self):
        phi = torch.rand(self.B, self.T, 1, requires_grad=True)
        psi = torch.rand(self.B, self.T, 1)
        conflict = torch.rand(self.B, self.T, 1)
        out = self.mediator(phi, psi, conflict)
        out.mediated_score.sum().backward()
        assert phi.grad is not None

    def test_param_count(self):
        n_params = sum(p.numel() for p in self.mediator.parameters())
        # mediator: 3*8+8+8*1+1 = 41
        assert 35 <= n_params <= 50


class TestMOPsiRegularization:
    """Testes da loss MOPsi."""

    def setup_method(self):
        self.loss_fn = MOPsiRegularization()

    def test_aligned_zero_loss(self):
        psi = torch.ones(2, 8, 1) * 0.7
        conf = torch.ones(2, 8, 1) * 0.7
        conflict = torch.zeros(2, 8, 1)
        loss = self.loss_fn(psi, conf, conflict)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_misaligned_positive_loss(self):
        psi = torch.ones(2, 8, 1) * 0.9
        conf = torch.ones(2, 8, 1) * 0.1
        conflict = torch.zeros(2, 8, 1)
        loss = self.loss_fn(psi, conf, conflict)
        assert loss.item() > 0.1

    def test_high_conflict_reduces_loss(self):
        psi = torch.ones(2, 8, 1) * 0.9
        conf = torch.ones(2, 8, 1) * 0.1
        low_conflict = torch.zeros(2, 8, 1)
        high_conflict = torch.ones(2, 8, 1) * 0.9
        loss_low = self.loss_fn(psi, conf, low_conflict)
        loss_high = self.loss_fn(psi, conf, high_conflict)
        assert loss_high < loss_low

    def test_gradient_flow(self):
        psi = torch.rand(2, 8, 1, requires_grad=True)
        conf = torch.rand(2, 8, 1)
        conflict = torch.rand(2, 8, 1)
        loss = self.loss_fn(psi, conf, conflict)
        loss.backward()
        assert psi.grad is not None
