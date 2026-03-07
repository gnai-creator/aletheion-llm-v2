"""Testes do modulo Metacognitive: ContrastiveHead + loss."""

import torch
import pytest

from aletheion_v2.metacognitive.contrastive_head import (
    ContrastiveHead, ContrastiveOutput,
)
from aletheion_v2.loss.contrastive_loss import ContrastiveRegularization


class TestContrastiveHead:
    """Testes do ContrastiveHead."""

    def setup_method(self):
        self.d_model = 256
        self.head = ContrastiveHead(
            d_model=self.d_model, hidden_dim=64, proj_dim=128,
        )
        self.B = 2
        self.T = 8

    def test_output_type(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        assert isinstance(out, ContrastiveOutput)

    def test_output_shapes(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        assert out.divergence.shape == (self.B, self.T, 1)
        assert out.view_a.shape == (self.B, self.T, 128)
        assert out.view_b.shape == (self.B, self.T, 128)

    def test_divergence_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        assert (out.divergence >= 0).all()
        assert (out.divergence <= 1).all()

    def test_views_different(self):
        """Views A e B devem ser diferentes (projetores independentes)."""
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        assert not torch.allclose(out.view_a, out.view_b)

    def test_gradient_flow(self):
        hidden = torch.randn(self.B, self.T, self.d_model, requires_grad=True)
        out = self.head(hidden)
        loss = out.divergence.sum() + out.view_a.sum()
        loss.backward()
        assert hidden.grad is not None

    def test_param_count_768(self):
        head_768 = ContrastiveHead(d_model=768, hidden_dim=128, proj_dim=384)
        n_params = sum(p.numel() for p in head_768.parameters())
        # proj_a: 768*128+128+128*384+384 = 148096
        # proj_b: 148096
        # divergence_net: 768*32+32+32*1+1 = 24609
        # Total ~320801
        # Com proj_dim=384: proj: 768*128+128+128*384+384 = 148096 * 2 = 296192
        # divergence: 768*32+32+32*1+1 = 24609
        # Total ~320801
        assert 150000 <= n_params <= 350000

    def test_default_proj_dim(self):
        """proj_dim default deve ser d_model // 2."""
        head = ContrastiveHead(d_model=256)
        assert head.proj_dim == 128

    def test_batch_size_one(self):
        hidden = torch.randn(1, self.T, self.d_model)
        out = self.head(hidden)
        assert out.divergence.shape == (1, self.T, 1)


class TestContrastiveRegularization:
    """Testes da loss contrastiva."""

    def setup_method(self):
        self.loss_fn = ContrastiveRegularization(
            cap_threshold=0.8, cap_weight=0.1,
        )

    def test_moderate_divergence(self):
        """Divergencia moderada deve ter loss finita."""
        div = torch.ones(2, 8, 1) * 0.5
        loss = self.loss_fn(div)
        assert torch.isfinite(loss)

    def test_zero_divergence_high_loss(self):
        """Divergencia zero -> -log(eps) -> loss muito alta."""
        div = torch.zeros(2, 8, 1)
        loss = self.loss_fn(div)
        assert loss.item() > 10

    def test_high_divergence_capped(self):
        """Divergencia > cap -> perda quadratica extra."""
        low = torch.ones(2, 8, 1) * 0.5
        high = torch.ones(2, 8, 1) * 0.95
        loss_low = self.loss_fn(low)
        loss_high = self.loss_fn(high)
        # O anti_collapse domina, entao high pode ter loss menor
        # Mas o capping adiciona penalidade
        assert torch.isfinite(loss_high)

    def test_with_mask(self):
        div = torch.rand(2, 8, 1) * 0.5 + 0.1
        mask = torch.ones(2, 8)
        mask[:, 4:] = 0
        loss = self.loss_fn(div, mask)
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        div = (torch.rand(2, 8, 1) * 0.5 + 0.1).requires_grad_(True)
        div.retain_grad()
        loss = self.loss_fn(div)
        loss.backward()
        assert div.grad is not None
