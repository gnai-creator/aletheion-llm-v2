"""Testes do modulo EidosDecay."""

import torch
import pytest

from aletheion_v2.eidos.eidos_decay import EidosDecay, EidosOutput
from aletheion_v2.loss.eidos_loss import EidosRegularization


class TestEidosDecay:
    """Testes do EidosDecay nn.Module."""

    def setup_method(self):
        self.eidos = EidosDecay(drm_dim=5)
        self.B = 2
        self.T = 8

    def test_output_type(self):
        coords = torch.rand(self.B, self.T, 5)
        conf = torch.rand(self.B, self.T, 1)
        out = self.eidos(coords, conf)
        assert isinstance(out, EidosOutput)

    def test_output_shapes(self):
        coords = torch.rand(self.B, self.T, 5)
        conf = torch.rand(self.B, self.T, 1)
        out = self.eidos(coords, conf)
        assert out.eidos_weights.shape == (self.B, self.T, 1)
        assert out.axis_balance.shape == (self.B, self.T, 5)
        assert out.axis_std.shape == (self.B, 5)

    def test_weights_range(self):
        coords = torch.rand(self.B, self.T, 5)
        conf = torch.rand(self.B, self.T, 1)
        out = self.eidos(coords, conf)
        assert (out.eidos_weights >= 0).all()
        assert (out.eidos_weights <= 1).all()

    def test_axis_balance_positive(self):
        coords = torch.rand(self.B, self.T, 5)
        conf = torch.rand(self.B, self.T, 1)
        out = self.eidos(coords, conf)
        assert (out.axis_balance > 0).all()

    def test_dream_mode_amplifies(self):
        coords = torch.rand(self.B, self.T, 5)
        conf = torch.ones(self.B, self.T, 1)
        out_normal = self.eidos(coords, conf, dream_mode=False)
        out_dream = self.eidos(coords, conf, dream_mode=True)
        # Dream mode deve amplificar desvio do 1.0
        dev_normal = (out_normal.axis_balance - 1.0).abs().mean()
        dev_dream = (out_dream.axis_balance - 1.0).abs().mean()
        assert dev_dream >= dev_normal

    def test_gradient_flow(self):
        coords = torch.rand(self.B, self.T, 5, requires_grad=True)
        conf = torch.rand(self.B, self.T, 1)
        out = self.eidos(coords, conf)
        loss = out.eidos_weights.sum()
        loss.backward()
        assert coords.grad is not None

    def test_param_count(self):
        n_params = sum(p.numel() for p in self.eidos.parameters())
        # decay_net: 5*16+16+16*5+5 = 181
        # reinforce_net: 181
        # weight_proj: 5*1+1 = 6
        # Total ~368
        assert 300 <= n_params <= 400

    def test_different_drm_dim(self):
        eidos3 = EidosDecay(drm_dim=3)
        coords = torch.rand(self.B, self.T, 3)
        conf = torch.rand(self.B, self.T, 1)
        out = eidos3(coords, conf)
        assert out.axis_balance.shape == (self.B, self.T, 3)


class TestEidosRegularization:
    """Testes da loss de regularizacao."""

    def setup_method(self):
        self.loss_fn = EidosRegularization(target_std=0.15)

    def test_uniform_balance_low_loss(self):
        # Balance uniforme -> std proximo de 0 -> loss = (0 - 0.15)^2
        balance = torch.ones(2, 8, 5)
        loss = self.loss_fn(balance)
        assert loss.item() < 0.1  # std=0, loss=(0-0.15)^2=0.0225

    def test_imbalanced_high_loss(self):
        # Balance desbalanceado -> std alta -> loss alta
        balance = torch.zeros(2, 8, 5)
        balance[:, :, 0] = 5.0
        loss = self.loss_fn(balance)
        assert loss.item() > 0.1

    def test_with_mask(self):
        balance = torch.rand(2, 8, 5)
        mask = torch.ones(2, 8)
        mask[:, 4:] = 0  # Metade e padding
        loss_masked = self.loss_fn(balance, mask)
        loss_full = self.loss_fn(balance)
        # Devem ser diferentes
        assert not torch.allclose(loss_masked, loss_full)

    def test_gradient_flow(self):
        balance = torch.rand(2, 8, 5, requires_grad=True)
        loss = self.loss_fn(balance)
        loss.backward()
        assert balance.grad is not None
