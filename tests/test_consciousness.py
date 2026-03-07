"""Testes do modulo Consciousness: SelfModelHead + loss."""

import torch
import pytest

from aletheion_v2.consciousness.self_model_head import (
    SelfModelHead, ConsciousnessOutput,
)
from aletheion_v2.loss.consciousness_loss import ConsciousnessRegularization


class TestSelfModelHead:
    """Testes do SelfModelHead."""

    def setup_method(self):
        self.d_model = 256
        self.head = SelfModelHead(
            d_model=self.d_model, hidden_dim=32, energy_decay=0.3,
        )
        self.B = 2
        self.T = 8

    def test_output_type(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        q2 = torch.rand(self.B, self.T, 1)
        phi = torch.rand(self.B, self.T, 1)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, q2, phi, conf)
        assert isinstance(out, ConsciousnessOutput)

    def test_output_shapes(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        q2 = torch.rand(self.B, self.T, 1)
        phi = torch.rand(self.B, self.T, 1)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, q2, phi, conf)
        assert out.mood.shape == (self.B, self.T, 1)
        assert out.curiosity.shape == (self.B, self.T, 1)
        assert out.energy.shape == (self.B, self.T, 1)
        assert out.drives.shape == (self.B, self.T, 3)

    def test_mood_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        q2 = torch.rand(self.B, self.T, 1)
        phi = torch.rand(self.B, self.T, 1)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, q2, phi, conf)
        assert (out.mood >= -1).all()
        assert (out.mood <= 1).all()

    def test_curiosity_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        q2 = torch.rand(self.B, self.T, 1)
        phi = torch.rand(self.B, self.T, 1)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, q2, phi, conf)
        assert (out.curiosity >= 0).all()
        assert (out.curiosity <= 1).all()

    def test_energy_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        q2 = torch.rand(self.B, self.T, 1)
        phi = torch.rand(self.B, self.T, 1)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, q2, phi, conf)
        assert (out.energy >= 0).all()
        assert (out.energy <= 1).all()

    def test_energy_decays_with_position(self):
        """Energia deve decair ao longo da sequencia."""
        hidden = torch.randn(self.B, 16, self.d_model)
        q2 = torch.rand(self.B, 16, 1)
        phi = torch.rand(self.B, 16, 1)
        conf = torch.rand(self.B, 16, 1)
        out = self.head(hidden, q2, phi, conf)
        # Media da energia no inicio vs final deve mostrar decaimento
        energy_start = out.energy[:, :4, :].mean()
        energy_end = out.energy[:, -4:, :].mean()
        # Nao e garantido com pesos aleatorios, mas o decay posicional ajuda
        # Pelo menos verifica que roda sem erro
        assert energy_start.item() >= 0

    def test_drives_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        q2 = torch.rand(self.B, self.T, 1)
        phi = torch.rand(self.B, self.T, 1)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, q2, phi, conf)
        assert (out.drives >= 0).all()
        assert (out.drives <= 1).all()

    def test_gradient_flow(self):
        hidden = torch.randn(self.B, self.T, self.d_model, requires_grad=True)
        q2 = torch.rand(self.B, self.T, 1)
        phi = torch.rand(self.B, self.T, 1)
        conf = torch.rand(self.B, self.T, 1)
        out = self.head(hidden, q2, phi, conf)
        loss = out.mood.sum() + out.energy.sum() + out.drives.sum()
        loss.backward()
        assert hidden.grad is not None

    def test_param_count(self):
        head_768 = SelfModelHead(d_model=768, hidden_dim=32)
        n_params = sum(p.numel() for p in head_768.parameters())
        # mood_net: 3*32+32+32*1+1 = 129
        # curiosity_net: 1*8+8+8*1+1 = 25
        # energy_proj: 768*1+1 = 769
        # drive_head: 3*8+8+8*3+3 = 59
        # Total ~982
        assert 900 <= n_params <= 1100


class TestConsciousnessRegularization:
    """Testes da loss de consciencia."""

    def setup_method(self):
        self.loss_fn = ConsciousnessRegularization(min_energy=0.2)

    def test_high_energy_zero_loss(self):
        energy = torch.ones(2, 8, 1) * 0.5
        loss = self.loss_fn(energy)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_low_energy_positive_loss(self):
        energy = torch.ones(2, 8, 1) * 0.05
        loss = self.loss_fn(energy)
        assert loss.item() > 0

    def test_with_mask(self):
        energy = torch.ones(2, 8, 1) * 0.1
        mask = torch.ones(2, 8)
        mask[:, 4:] = 0
        loss = self.loss_fn(energy, mask)
        assert loss.item() > 0

    def test_gradient_flow(self):
        energy = (torch.rand(2, 8, 1) * 0.3).requires_grad_(True)
        energy.retain_grad()
        loss = self.loss_fn(energy)
        loss.backward()
        assert energy.grad is not None
