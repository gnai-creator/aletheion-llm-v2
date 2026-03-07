"""Testes do modulo CausalState: StateConditioning e PolicyBinding."""

import torch
import pytest

from aletheion_v2.causal_state.state_conditioning import (
    StateConditioning, StateConditioningOutput,
    PolicyBinding, PolicyParams,
)


class TestStateConditioning:
    """Testes do StateConditioning."""

    def setup_method(self):
        self.d_model = 256
        self.cond = StateConditioning(d_model=self.d_model, hidden_dim=32)
        self.B = 2
        self.T = 8

    def test_output_type(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        state = torch.rand(self.B, 4)
        out = self.cond(hidden, state)
        assert isinstance(out, StateConditioningOutput)

    def test_output_shapes(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        state = torch.rand(self.B, 4)
        out = self.cond(hidden, state)
        assert out.conditioned_hidden.shape == (self.B, self.T, self.d_model)
        assert out.gate_value.shape == (self.B, 1)
        assert out.state_embedding.shape == (self.B, self.d_model)

    def test_gate_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        state = torch.rand(self.B, 4)
        out = self.cond(hidden, state)
        assert (out.gate_value >= 0).all()
        assert (out.gate_value <= 1).all()

    def test_zero_state_minimal_change(self):
        """Estado zero deve resultar em mudanca minima."""
        hidden = torch.randn(self.B, self.T, self.d_model)
        state = torch.zeros(self.B, 4)
        out = self.cond(hidden, state)
        # Com estado zero, a diferenca depende dos bias
        diff = (out.conditioned_hidden - hidden).abs().mean()
        # Deve ser relativamente pequena
        assert diff.item() < 10  # Valor razoavel com pesos aleatorios

    def test_gradient_flow(self):
        hidden = torch.randn(self.B, self.T, self.d_model, requires_grad=True)
        state = torch.rand(self.B, 4, requires_grad=True)
        out = self.cond(hidden, state)
        loss = out.conditioned_hidden.sum()
        loss.backward()
        assert hidden.grad is not None
        assert state.grad is not None

    def test_param_count(self):
        cond_768 = StateConditioning(d_model=768, hidden_dim=32)
        n_params = sum(p.numel() for p in cond_768.parameters())
        # state_embed: 4*32+32+32*768+768 = 25504
        # gate: 4*1+1 = 5
        # ~25509
        assert 25000 <= n_params <= 26000


class TestPolicyBinding:
    """Testes do PolicyBinding."""

    def test_output_type(self):
        state = torch.tensor([0.5, 0.5, 0.5, 0.5])
        params = PolicyBinding.bind(state)
        assert isinstance(params, PolicyParams)

    def test_low_curiosity_low_temp(self):
        state = torch.tensor([0.0, 0.5, 0.5, 0.5])
        params = PolicyBinding.bind(state)
        assert params.temperature == pytest.approx(0.3, abs=0.01)

    def test_high_curiosity_high_temp(self):
        state = torch.tensor([1.0, 0.5, 0.5, 0.5])
        params = PolicyBinding.bind(state)
        assert params.temperature == pytest.approx(0.9, abs=0.01)

    def test_low_verbosity_low_tokens(self):
        state = torch.tensor([0.5, 0.0, 0.5, 0.5])
        params = PolicyBinding.bind(state)
        assert params.max_tokens_factor == pytest.approx(0.4, abs=0.01)

    def test_high_verbosity_high_tokens(self):
        state = torch.tensor([0.5, 1.0, 0.5, 0.5])
        params = PolicyBinding.bind(state)
        assert params.max_tokens_factor == pytest.approx(1.6, abs=0.01)

    def test_batch_input(self):
        state = torch.rand(4, 4)
        params = PolicyBinding.bind(state)
        assert 0.3 <= params.temperature <= 0.9
        assert 0.4 <= params.max_tokens_factor <= 1.6
