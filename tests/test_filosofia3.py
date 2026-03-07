"""Testes do modulo Filosofia3: deteccao de conflito phi-psi."""

import torch
import pytest

from aletheion_v2.filosofia3.conflict_head import PhiPsiConflictHead, ConflictOutput
from aletheion_v2.loss.conflict_loss import ConflictRegularization


class TestPhiPsiConflictHead:
    """Testes do PhiPsiConflictHead."""

    def setup_method(self):
        self.head = PhiPsiConflictHead()
        self.B = 2
        self.T = 8

    def test_output_type(self):
        phi = torch.rand(self.B, self.T, 4)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(phi, conf)
        assert isinstance(out, ConflictOutput)

    def test_output_shapes(self):
        phi = torch.rand(self.B, self.T, 4)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(phi, conf)
        assert out.conflict_intensity.shape == (self.B, self.T, 1)
        assert out.mode_probs.shape == (self.B, self.T, 4)
        assert out.conflict_analytical.shape == (self.B, self.T, 1)

    def test_conflict_range(self):
        phi = torch.rand(self.B, self.T, 4)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(phi, conf)
        assert (out.conflict_intensity >= 0).all()
        assert (out.conflict_intensity <= 1).all()

    def test_mode_probs_sum_to_one(self):
        phi = torch.rand(self.B, self.T, 4)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(phi, conf)
        sums = out.mode_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_constant_phi_low_conflict(self):
        """Phi constante entre tokens -> deltas zero -> conflito baixo."""
        phi = torch.ones(self.B, self.T, 4) * 0.5
        conf = torch.ones(self.B, self.T, 1) * 0.5
        out = self.head(phi, conf)
        # Conflito analitico deve ser baixo (deltas zero -> cos indefinido -> 0.5)
        # mas aprendivel pode variar
        assert out.conflict_intensity.mean().item() < 0.8

    def test_single_token(self):
        """T=1 nao deve quebrar."""
        phi = torch.rand(self.B, 1, 4)
        conf = torch.rand(self.B, 1, 1)
        out = self.head(phi, conf)
        assert out.conflict_intensity.shape == (self.B, 1, 1)

    def test_gradient_flow(self):
        phi = torch.rand(self.B, self.T, 4, requires_grad=True)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(phi, conf)
        loss = out.conflict_intensity.sum()
        loss.backward()
        assert phi.grad is not None

    def test_param_count(self):
        n_params = sum(p.numel() for p in self.head.parameters())
        # conflict_proj: 8*16+16+16*1+1 = 161
        # mode_head: 8*16+16+16*4+4 = 212
        # Total ~373
        assert 300 <= n_params <= 450

    def test_analytical_weight(self):
        """Componente analitica deve dominar com peso 0.7."""
        head_full_analytical = PhiPsiConflictHead(analytical_weight=1.0)
        phi = torch.rand(self.B, self.T, 4)
        conf = torch.rand(self.B, self.T, 1)
        out = head_full_analytical(phi, conf)
        # Deve ser identico ao analitico
        assert torch.allclose(
            out.conflict_intensity, out.conflict_analytical, atol=1e-5
        )


class TestConflictRegularization:
    """Testes da loss de conflito."""

    def setup_method(self):
        self.loss_fn = ConflictRegularization()

    def test_zero_conflict_zero_loss(self):
        conflict = torch.zeros(2, 8, 1)
        loss = self.loss_fn(conflict)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_high_conflict_high_loss(self):
        conflict = torch.ones(2, 8, 1)
        loss = self.loss_fn(conflict)
        assert loss.item() > 0.5

    def test_with_mask(self):
        conflict = torch.rand(2, 8, 1)
        mask = torch.ones(2, 8)
        mask[:, 4:] = 0
        loss = self.loss_fn(conflict, mask)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        conflict = torch.rand(2, 8, 1, requires_grad=True)
        loss = self.loss_fn(conflict)
        loss.backward()
        assert conflict.grad is not None
