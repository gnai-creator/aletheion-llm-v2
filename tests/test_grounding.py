"""Testes dos modulos Grounding: TaskClassificationHead e AmbiguityHead."""

import torch
import pytest

from aletheion_v2.grounding.task_head import TaskClassificationHead, TaskOutput
from aletheion_v2.grounding.ambiguity_head import AmbiguityHead, AmbiguityOutput
from aletheion_v2.loss.grounding_loss import GroundingRegularization


class TestTaskClassificationHead:
    """Testes do TaskClassificationHead."""

    def setup_method(self):
        self.d_model = 256
        self.head = TaskClassificationHead(d_model=self.d_model)
        self.B = 2
        self.T = 8

    def test_output_type(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        assert isinstance(out, TaskOutput)

    def test_output_shapes(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        assert out.task_probs.shape == (self.B, self.T, 9)
        assert out.task_confidence.shape == (self.B, self.T, 1)

    def test_probs_sum_to_one(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        sums = out.task_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_confidence_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        assert (out.task_confidence >= 0).all()
        assert (out.task_confidence <= 1).all()

    def test_gradient_flow(self):
        hidden = torch.randn(self.B, self.T, self.d_model, requires_grad=True)
        out = self.head(hidden)
        loss = out.task_probs.sum() + out.task_confidence.sum()
        loss.backward()
        assert hidden.grad is not None

    def test_param_count(self):
        head_768 = TaskClassificationHead(d_model=768, hidden_dim=64)
        n_params = sum(p.numel() for p in head_768.parameters())
        # classifier: 768*64+64+64*9+9 = 49921
        # conf_proj: 9*1+1 = 10
        # ~49931
        assert 49000 <= n_params <= 51000


class TestAmbiguityHead:
    """Testes do AmbiguityHead."""

    def setup_method(self):
        self.d_model = 256
        self.head = AmbiguityHead(d_model=self.d_model)
        self.B = 2
        self.T = 8

    def test_output_type(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        assert isinstance(out, AmbiguityOutput)

    def test_output_shapes(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        assert out.ambiguity_level.shape == (self.B, self.T, 1)
        assert out.ambiguity_type.shape == (self.B, self.T, 5)

    def test_level_range(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        assert (out.ambiguity_level >= 0).all()
        assert (out.ambiguity_level <= 1).all()

    def test_type_probs_sum_to_one(self):
        hidden = torch.randn(self.B, self.T, self.d_model)
        out = self.head(hidden)
        sums = out.ambiguity_type.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flow(self):
        hidden = torch.randn(self.B, self.T, self.d_model, requires_grad=True)
        out = self.head(hidden)
        loss = out.ambiguity_level.sum()
        loss.backward()
        assert hidden.grad is not None

    def test_param_count(self):
        head_768 = AmbiguityHead(d_model=768, hidden_dim=32)
        n_params = sum(p.numel() for p in head_768.parameters())
        # level_net: 768*32+32+32*1+1 = 24609
        # type_net: 768*32+32+32*5+5 = 24741
        # ~49350
        assert 48000 <= n_params <= 51000


class TestGroundingRegularization:
    """Testes da loss de grounding."""

    def setup_method(self):
        self.loss_fn = GroundingRegularization()

    def test_decisive_classification_low_entropy(self):
        # One-hot probs -> entropia zero
        probs = torch.zeros(2, 8, 9)
        probs[:, :, 0] = 1.0
        ambiguity = torch.zeros(2, 8, 1)
        q1 = torch.zeros(2, 8, 1)
        loss = self.loss_fn(probs, ambiguity, q1)
        assert loss.item() < 0.01

    def test_uniform_classification_high_entropy(self):
        probs = torch.ones(2, 8, 9) / 9
        ambiguity = torch.zeros(2, 8, 1)
        q1 = torch.zeros(2, 8, 1)
        loss = self.loss_fn(probs, ambiguity, q1)
        assert loss.item() > 0.5

    def test_calibrated_ambiguity_low_loss(self):
        probs = torch.ones(2, 8, 9) / 9  # Uniforme
        q1 = torch.ones(2, 8, 1) * 0.5
        ambiguity = torch.ones(2, 8, 1) * 0.5  # Alinhado com q1
        loss = self.loss_fn(probs, ambiguity, q1)
        # Calibracao = 0, so entropia
        loss_entropy_only = self.loss_fn(probs, q1, q1)
        assert torch.allclose(loss, loss_entropy_only, atol=1e-5)

    def test_with_mask(self):
        probs = torch.ones(2, 8, 9) / 9
        ambiguity = torch.rand(2, 8, 1)
        q1 = torch.rand(2, 8, 1)
        mask = torch.ones(2, 8)
        mask[:, 4:] = 0
        loss = self.loss_fn(probs, ambiguity, q1, mask)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        probs = torch.rand(2, 8, 9, requires_grad=True)
        probs_norm = torch.softmax(probs, dim=-1)
        ambiguity = torch.rand(2, 8, 1)
        q1 = torch.rand(2, 8, 1)
        loss = self.loss_fn(probs_norm, ambiguity, q1)
        loss.backward()
        assert probs.grad is not None
