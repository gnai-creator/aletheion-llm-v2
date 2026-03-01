"""
Testes do modelo AletheionV2Model completo.
"""

import torch
import pytest

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model
from aletheion_v2.core.output import ModelOutput, EpistemicTomography
from aletheion_v2.loss.composite_loss import AletheionV2Loss


class TestAletheionV2Config:
    """Testes de configuracao."""

    def test_small_config(self):
        config = AletheionV2Config.small()
        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.n_layers == 4

    def test_medium_config(self):
        config = AletheionV2Config.medium()
        assert config.d_model == 768
        assert config.n_heads == 12

    def test_head_dim(self):
        config = AletheionV2Config.small()
        assert config.head_dim == 64  # 256 / 4

    def test_yaml_roundtrip(self, tmp_path):
        config = AletheionV2Config.small()
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)
        loaded = AletheionV2Config.from_yaml(path)
        assert loaded.d_model == config.d_model
        assert loaded.n_heads == config.n_heads


class TestAletheionV2Model:
    """Testes do modelo completo."""

    def setup_method(self):
        self.config = AletheionV2Config.small()
        self.model = AletheionV2Model(self.config)
        self.B = 2
        self.T = 16

    def test_forward_shapes(self):
        input_ids = torch.randint(0, self.config.vocab_size, (self.B, self.T))
        output = self.model(input_ids)
        assert output.logits.shape == (self.B, self.T, self.config.vocab_size)
        assert output.hidden_states.shape == (self.B, self.T, self.config.d_model)

    def test_forward_with_tomography(self):
        input_ids = torch.randint(0, self.config.vocab_size, (self.B, self.T))
        output = self.model(input_ids, return_tomography=True)
        assert output.tomography is not None
        assert output.tomography.q1.shape == (self.B, self.T, 1)
        assert output.tomography.drm_coords.shape == (self.B, self.T, 5)

    def test_forward_without_tomography(self):
        input_ids = torch.randint(0, self.config.vocab_size, (self.B, self.T))
        output = self.model(input_ids, return_tomography=False)
        assert output.tomography is None
        assert output.attention_patterns is None

    def test_gradient_flow(self):
        input_ids = torch.randint(0, self.config.vocab_size, (self.B, self.T))
        output = self.model(input_ids, return_tomography=True)
        loss = output.logits.sum() + output.tomography.confidence.sum()
        loss.backward()
        # Verifica que gradientes fluem ate embeddings
        assert self.model.embeddings.token_emb.weight.grad is not None

    def test_causal_mask(self):
        """Cada posicao so ve tokens anteriores."""
        input_ids = torch.randint(0, self.config.vocab_size, (1, self.T))
        output = self.model(input_ids, return_tomography=True)
        # Verifica attention patterns
        attn = output.attention_patterns  # [1, L, H, T, T]
        # Mascara causal: posicoes futuras devem ter atencao ~0
        for t in range(self.T):
            future_attn = attn[0, 0, 0, t, t + 1 :]
            assert (future_attn < 1e-5).all()

    def test_count_parameters(self):
        counts = self.model.count_parameters()
        assert "total" in counts
        assert "epistemic_head" in counts
        assert "epistemic_pct" in counts
        assert counts["total"] > 0
        # Epistemic head deve ser ~1-5% do total
        assert counts["epistemic_pct"] < 10

    def test_weight_tying(self):
        """lm_head.weight deve ser o mesmo que token_emb.weight."""
        assert self.model.lm_head.weight is self.model.embeddings.token_emb.weight

    def test_generate_next_token(self):
        input_ids = torch.randint(0, self.config.vocab_size, (1, self.T))
        next_token, tomo = self.model.generate_next_token(input_ids)
        assert next_token.shape == (1, 1)
        assert 0 <= next_token.item() < self.config.vocab_size


class TestAletheionV2Loss:
    """Testes da loss composta."""

    def setup_method(self):
        self.config = AletheionV2Config.small()
        self.model = AletheionV2Model(self.config)
        self.loss_fn = AletheionV2Loss(self.config)

    def test_ce_only(self):
        """Sem tomografia, so CE."""
        logits = torch.randn(2, 8, self.config.vocab_size)
        labels = torch.randint(0, self.config.vocab_size, (2, 8))
        losses = self.loss_fn(logits, labels)
        assert "total" in losses
        assert "ce" in losses
        assert losses["total"].item() > 0

    def test_full_loss(self):
        """Com tomografia, todas as componentes."""
        input_ids = torch.randint(0, self.config.vocab_size, (2, 8))
        labels = torch.randint(0, self.config.vocab_size, (2, 8))
        output = self.model(input_ids, return_tomography=True)

        G = self.model.epistemic_head.get_metric_tensor()
        losses = self.loss_fn(
            output.logits, labels,
            tomography=output.tomography,
            G=G,
            step=1000,
            total_steps=2000,
        )

        assert all(
            k in losses
            for k in ["total", "ce", "varo", "vi", "mad", "metric", "annealing"]
        )
        assert losses["total"].item() > 0

    def test_annealing(self):
        """Warmup deve ter anneal = 0, apos ramp deve ter anneal = 1."""
        logits = torch.randn(2, 8, self.config.vocab_size)
        labels = torch.randint(0, self.config.vocab_size, (2, 8))

        # Step 0 (warmup)
        losses_early = self.loss_fn(logits, labels, step=0, total_steps=10000)
        assert losses_early["annealing"].item() == 0.0

        # Step 9999 (final)
        losses_late = self.loss_fn(logits, labels, step=9999, total_steps=10000)
        assert losses_late["annealing"].item() == 1.0

    def test_gradient_backward(self):
        """Loss deve permitir backward para maioria dos parametros."""
        input_ids = torch.randint(0, self.config.vocab_size, (2, 8))
        labels = torch.randint(0, self.config.vocab_size, (2, 8))
        output = self.model(input_ids, return_tomography=True)
        G = self.model.epistemic_head.get_metric_tensor()

        losses = self.loss_fn(
            output.logits, labels,
            tomography=output.tomography,
            G=G,
            step=500,
            total_steps=1000,
        )
        losses["total"].backward()

        # Verifica que maioria dos parametros tem gradiente
        # directional_field nao participa de nenhuma loss diretamente
        trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        with_grad = sum(
            1 for p in self.model.parameters()
            if p.requires_grad and p.grad is not None
        )
        ratio = with_grad / trainable
        assert ratio > 0.8, f"Apenas {with_grad}/{trainable} params com grad"

    def test_metric_regularization(self):
        G = torch.eye(5)
        reg = self.loss_fn.metric_regularization(G)
        # G = I -> kappa = 1 -> log(2)
        assert reg.item() > 0
