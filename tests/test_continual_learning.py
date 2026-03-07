"""Testes do modulo de Continual Learning: EWC + Replay Buffer."""

import torch
import pytest

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model
from aletheion_v2.training.ewc import EWCRegularizer
from aletheion_v2.training.replay_buffer import ReplayBuffer
from aletheion_v2.training.data import create_synthetic_data


# --- EWC ---


class TestEWCRegularizer:
    """Testes do EWCRegularizer."""

    def setup_method(self):
        self.config = AletheionV2Config.small()
        self.model = AletheionV2Model(self.config)

    def test_init_no_fisher(self):
        ewc = EWCRegularizer(lambda_ewc=100.0)
        assert not ewc.has_fisher
        loss = ewc(self.model)
        assert loss.item() == 0.0

    def test_compute_fisher(self):
        ewc = EWCRegularizer(lambda_ewc=100.0)
        loader = create_synthetic_data(
            vocab_size=self.config.vocab_size,
            total_tokens=256,
            seq_len=16,
            batch_size=4,
        )
        ewc.compute_fisher(
            self.model, loader, num_samples=8, device="cpu",
        )
        assert ewc.has_fisher
        assert ewc._num_phases == 1

    def test_fisher_positive(self):
        """Fisher diagonal deve ser >= 0."""
        ewc = EWCRegularizer()
        loader = create_synthetic_data(
            vocab_size=self.config.vocab_size,
            total_tokens=256,
            seq_len=16,
            batch_size=4,
        )
        ewc.compute_fisher(self.model, loader, num_samples=8)
        for name, fisher in ewc._fisher.items():
            assert (fisher >= 0).all(), f"Fisher negativa em {name}"

    def test_ewc_loss_nonzero_after_change(self):
        """EWC loss deve ser > 0 se parametros mudaram."""
        ewc = EWCRegularizer(lambda_ewc=100.0)
        loader = create_synthetic_data(
            vocab_size=self.config.vocab_size,
            total_tokens=256,
            seq_len=16,
            batch_size=4,
        )
        ewc.compute_fisher(self.model, loader, num_samples=8)

        # Altera parametros
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        loss = ewc(self.model)
        assert loss.item() > 0.0

    def test_ewc_loss_zero_no_change(self):
        """EWC loss deve ser ~0 se parametros nao mudaram."""
        ewc = EWCRegularizer(lambda_ewc=100.0)
        loader = create_synthetic_data(
            vocab_size=self.config.vocab_size,
            total_tokens=256,
            seq_len=16,
            batch_size=4,
        )
        ewc.compute_fisher(self.model, loader, num_samples=8)
        loss = ewc(self.model)
        assert loss.item() < 1e-6

    def test_gradient_flow(self):
        """EWC loss deve permitir backward."""
        ewc = EWCRegularizer(lambda_ewc=100.0)
        loader = create_synthetic_data(
            vocab_size=self.config.vocab_size,
            total_tokens=256,
            seq_len=16,
            batch_size=4,
        )
        ewc.compute_fisher(self.model, loader, num_samples=8)

        # Altera parametros para ter loss > 0
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        loss = ewc(self.model)
        loss.backward()

        has_grad = any(
            p.grad is not None
            for p in self.model.parameters()
            if p.requires_grad
        )
        assert has_grad

    def test_online_ewc_accumulates(self):
        """Online EWC deve acumular Fisher de multiplas fases."""
        ewc = EWCRegularizer(lambda_ewc=100.0, online=True, gamma=0.9)
        loader = create_synthetic_data(
            vocab_size=self.config.vocab_size,
            total_tokens=256,
            seq_len=16,
            batch_size=4,
        )

        ewc.compute_fisher(self.model, loader, num_samples=8)
        assert ewc._num_phases == 1

        ewc.compute_fisher(self.model, loader, num_samples=8)
        assert ewc._num_phases == 2

    def test_state_dict_roundtrip(self):
        """State dict deve preservar Fisher e params."""
        ewc = EWCRegularizer(lambda_ewc=100.0)
        loader = create_synthetic_data(
            vocab_size=self.config.vocab_size,
            total_tokens=256,
            seq_len=16,
            batch_size=4,
        )
        ewc.compute_fisher(self.model, loader, num_samples=8)

        state = ewc.state_dict_ewc()
        ewc2 = EWCRegularizer(lambda_ewc=100.0)
        ewc2.load_state_dict_ewc(state)

        assert ewc2.has_fisher
        assert ewc2._num_phases == 1

    def test_importance_stats(self):
        ewc = EWCRegularizer()
        loader = create_synthetic_data(
            vocab_size=self.config.vocab_size,
            total_tokens=256,
            seq_len=16,
            batch_size=4,
        )
        ewc.compute_fisher(self.model, loader, num_samples=8)
        stats = ewc.get_importance_stats()
        assert "fisher/total_importance" in stats
        assert "fisher/avg_importance" in stats
        assert stats["fisher/num_phases"] == 1


# --- Replay Buffer ---


class TestReplayBuffer:
    """Testes do ReplayBuffer."""

    def test_empty_buffer(self):
        buf = ReplayBuffer(buffer_size=100)
        assert buf.is_empty
        assert buf.size == 0
        assert buf.sample(4) is None

    def test_add_samples(self):
        buf = ReplayBuffer(buffer_size=100)
        batch = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "labels": torch.randint(0, 100, (8, 16)),
        }
        buf.add(batch)
        assert buf.size == 8
        assert not buf.is_empty

    def test_sample_returns_correct_shape(self):
        buf = ReplayBuffer(buffer_size=100)
        batch = {
            "input_ids": torch.randint(0, 100, (16, 32)),
            "labels": torch.randint(0, 100, (16, 32)),
        }
        buf.add(batch)

        sample = buf.sample(4)
        assert sample is not None
        assert sample["input_ids"].shape == (4, 32)
        assert sample["labels"].shape == (4, 32)

    def test_reservoir_sampling_respects_max(self):
        """Buffer nao deve exceder buffer_size."""
        buf = ReplayBuffer(buffer_size=20)
        for _ in range(10):
            batch = {
                "input_ids": torch.randint(0, 100, (8, 16)),
                "labels": torch.randint(0, 100, (8, 16)),
            }
            buf.add(batch)

        assert buf.size == 20  # Max 20
        assert buf._n_seen == 80  # 10 * 8

    def test_mix_batch_replaces_samples(self):
        """mix_batch deve substituir parte do batch."""
        buf = ReplayBuffer(buffer_size=100, mix_ratio=0.5)
        # Preenche buffer com zeros
        zeros_batch = {
            "input_ids": torch.zeros(16, 32, dtype=torch.long),
            "labels": torch.zeros(16, 32, dtype=torch.long),
        }
        buf.add(zeros_batch)

        # Batch atual com uns
        ones_batch = {
            "input_ids": torch.ones(16, 32, dtype=torch.long),
            "labels": torch.ones(16, 32, dtype=torch.long),
        }

        mixed = buf.mix_batch(ones_batch)
        assert mixed["input_ids"].shape == (16, 32)
        # Deve ter mix de zeros e uns
        has_zeros = (mixed["input_ids"] == 0).any()
        has_ones = (mixed["input_ids"] == 1).any()
        assert has_zeros and has_ones

    def test_mix_batch_empty_buffer_passthrough(self):
        """Buffer vazio deve retornar batch original."""
        buf = ReplayBuffer(buffer_size=100, mix_ratio=0.5)
        batch = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "labels": torch.randint(0, 100, (8, 16)),
        }
        mixed = buf.mix_batch(batch)
        assert torch.equal(mixed["input_ids"], batch["input_ids"])

    def test_mix_batch_different_seq_len(self):
        """Deve lidar com seq_len diferentes (trunca/pad)."""
        buf = ReplayBuffer(buffer_size=100, mix_ratio=0.5)
        # Buffer com seq_len=32
        buf.add({
            "input_ids": torch.randint(0, 100, (8, 32)),
            "labels": torch.randint(0, 100, (8, 32)),
        })
        # Batch atual com seq_len=16
        batch = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "labels": torch.randint(0, 100, (8, 16)),
        }
        mixed = buf.mix_batch(batch)
        assert mixed["input_ids"].shape[1] == 16  # Truncado

    def test_clear(self):
        buf = ReplayBuffer(buffer_size=100)
        buf.add({
            "input_ids": torch.randint(0, 100, (8, 16)),
            "labels": torch.randint(0, 100, (8, 16)),
        })
        assert buf.size == 8
        buf.clear()
        assert buf.is_empty
        assert buf._n_seen == 0

    def test_state_dict_roundtrip(self):
        buf = ReplayBuffer(buffer_size=100)
        buf.add({
            "input_ids": torch.randint(0, 100, (8, 16)),
            "labels": torch.randint(0, 100, (8, 16)),
        })
        state = buf.state_dict()

        buf2 = ReplayBuffer(buffer_size=100)
        buf2.load_state_dict(state)
        assert buf2.size == 8
        assert buf2._n_seen == 8

    def test_stats(self):
        buf = ReplayBuffer(buffer_size=100, mix_ratio=0.1)
        buf.add({
            "input_ids": torch.randint(0, 100, (8, 16)),
            "labels": torch.randint(0, 100, (8, 16)),
        })
        stats = buf.stats()
        assert stats["replay/buffer_size"] == 8
        assert stats["replay/max_size"] == 100
        assert stats["replay/fill_ratio"] == 0.08
        assert stats["replay/total_seen"] == 8
        assert stats["replay/mix_ratio"] == 0.1


# --- Config CL ---


class TestCLConfig:
    """Testes dos campos de Continual Learning no config."""

    def test_cl_defaults_off(self):
        config = AletheionV2Config.small()
        assert config.enable_ewc is False
        assert config.enable_replay is False

    def test_cl_fields_exist(self):
        config = AletheionV2Config.small()
        assert hasattr(config, "ewc_lambda")
        assert hasattr(config, "ewc_online")
        assert hasattr(config, "ewc_gamma")
        assert hasattr(config, "ewc_fisher_samples")
        assert hasattr(config, "replay_buffer_size")
        assert hasattr(config, "replay_mix_ratio")

    def test_cl_yaml_roundtrip(self, tmp_path):
        config = AletheionV2Config.small()
        config.enable_ewc = True
        config.ewc_lambda = 200.0
        config.enable_replay = True
        config.replay_buffer_size = 5000

        path = str(tmp_path / "cl_config.yaml")
        config.to_yaml(path)
        loaded = AletheionV2Config.from_yaml(path)

        assert loaded.enable_ewc is True
        assert loaded.ewc_lambda == 200.0
        assert loaded.enable_replay is True
        assert loaded.replay_buffer_size == 5000

    def test_rtx4090_config_loads(self):
        """Config 350m_rtx4090.yaml deve carregar."""
        import os
        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "..", "configs", "scaling", "350m_rtx4090.yaml",
        )
        if os.path.exists(yaml_path):
            config = AletheionV2Config.from_yaml(yaml_path)
            assert config.d_model == 1024
            assert config.enable_ewc is True
            assert config.enable_replay is True
            assert config.gradient_checkpointing is True
            assert config.mixed_precision == "bf16"


# --- Scaling Configs ---


class TestScalingConfigs:
    """Verifica que configs de escala carregam corretamente."""

    CONFIGS = [
        ("1m", 64, 2, 4),
        ("10m", 256, 4, 6),
        ("50m", 512, 8, 8),
        ("125m", 768, 12, 12),
        ("350m", 1024, 16, 24),
        ("1.3b", 2048, 16, 24),
        ("7b", 4096, 32, 32),
        ("13b", 5120, 40, 40),
        ("30b", 6656, 52, 60),
        ("70b", 8192, 64, 80),
        ("162b", 12288, 96, 96),
        ("250b", 16384, 128, 80),
        ("400b", 18432, 144, 96),
        ("640b", 20480, 160, 128),
    ]

    def _get_config_path(self, name):
        import os
        return os.path.join(
            os.path.dirname(__file__),
            "..", "configs", "scaling", f"{name}.yaml",
        )

    @pytest.mark.parametrize("name,d_model,n_heads,n_layers", CONFIGS)
    def test_config_loads(self, name, d_model, n_heads, n_layers):
        """Cada config deve carregar com valores corretos."""
        import os
        path = self._get_config_path(name)
        if not os.path.exists(path):
            pytest.skip(f"Config {name}.yaml nao encontrada")
        config = AletheionV2Config.from_yaml(path)
        assert config.d_model == d_model
        assert config.n_heads == n_heads
        assert config.n_layers == n_layers

    @pytest.mark.parametrize("name,d_model,n_heads,n_layers", CONFIGS)
    def test_head_dim_divisible(self, name, d_model, n_heads, n_layers):
        """d_model deve ser divisivel por n_heads."""
        assert d_model % n_heads == 0, (
            f"{name}: {d_model} % {n_heads} = {d_model % n_heads}"
        )
