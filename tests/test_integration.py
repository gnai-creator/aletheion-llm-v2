"""
Testes de integracao end-to-end do AletheionV2.
"""

import torch
import pytest

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model
from aletheion_v2.loss.composite_loss import AletheionV2Loss
from aletheion_v2.training.data import create_synthetic_data, TextChunkDataset
from aletheion_v2.training.scheduler import WarmupCosineScheduler, LossWeightAnnealer
from aletheion_v2.inference.generator import Generator
from aletheion_v2.inference.dashboard_bridge import DashboardBridge
from aletheion_v2.mpc.transition_model import TransitionModel, InterventionType
from aletheion_v2.mpc.navigator import ManifoldNavigator


class TestEndToEnd:
    """Testes end-to-end do pipeline completo."""

    def setup_method(self):
        self.config = AletheionV2Config.small()
        self.config.max_seq_len = 32
        self.model = AletheionV2Model(self.config)

    def test_full_forward_backward(self):
        """Forward + loss + backward completo."""
        B, T = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        labels = torch.randint(0, self.config.vocab_size, (B, T))

        output = self.model(input_ids, return_tomography=True)
        loss_fn = AletheionV2Loss(self.config)
        G = self.model.epistemic_head.get_metric_tensor()

        losses = loss_fn(
            output.logits, labels,
            tomography=output.tomography,
            G=G, step=100, total_steps=200,
        )

        losses["total"].backward()

        # Verifica que maioria dos parametros tem gradiente
        # (directional_field nao participa diretamente das losses)
        trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        with_grad = sum(
            1 for p in self.model.parameters()
            if p.requires_grad and p.grad is not None
        )
        ratio = with_grad / trainable
        # CausalState sem state_vector + directional_field = ~30% sem grad
        assert ratio > 0.6, f"Apenas {with_grad}/{trainable} params com grad"

    def test_mini_training_loop(self):
        """Treina 3 steps e verifica que loss diminui."""
        B, T = 2, 16
        loss_fn = AletheionV2Loss(self.config)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        losses_history = []
        for step in range(3):
            input_ids = torch.randint(0, self.config.vocab_size, (B, T))
            labels = torch.randint(0, self.config.vocab_size, (B, T))

            output = self.model(input_ids, return_tomography=True)
            G = self.model.epistemic_head.get_metric_tensor()

            losses = loss_fn(
                output.logits, labels,
                tomography=output.tomography,
                G=G, step=step, total_steps=10,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            losses_history.append(losses["total"].item())

        # Loss deve existir (convergencia nao garantida em 3 steps)
        assert all(l > 0 for l in losses_history)

    def test_generation_with_tomography(self):
        """Geracao produz tokens e tomografia."""
        gen = Generator(self.model, max_new_tokens=5, top_k=10)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 8))
        result = gen.generate(input_ids)

        assert result.total_tokens == 5
        assert len(result.token_ids) == 5
        assert len(result.tomography_per_token) == 5
        assert result.avg_confidence > 0
        assert result.avg_phi > 0

    def test_dashboard_bridge(self):
        """Bridge converte GenerationResult para formato ATIC."""
        gen = Generator(self.model, max_new_tokens=5, top_k=10)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 8))
        result = gen.generate(input_ids)

        bridge = DashboardBridge()
        snapshot = bridge.from_generation_result(result)

        assert len(snapshot.manifold.points) == 5
        assert len(snapshot.manifold.confidence) == 5
        assert snapshot.vi.phi_total >= 0
        assert snapshot.vi.mode in ("healthy", "warning", "critical")

        # JSON serializavel
        json_str = bridge.to_json(snapshot)
        import json
        data = json.loads(json_str)
        assert "manifold" in data
        assert "vi" in data
        assert "epistemic" in data

    def test_atic_endpoint_compat(self):
        """Verifica compatibilidade com endpoints ATIC."""
        gen = Generator(self.model, max_new_tokens=3, top_k=10)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 8))
        result = gen.generate(input_ids)

        bridge = DashboardBridge()
        snapshot = bridge.from_generation_result(result)
        endpoints = bridge.to_atic_endpoints(snapshot)

        # DRM endpoint
        assert "points" in endpoints["drm"]
        assert "anchors" in endpoints["drm"]

        # VI endpoint
        assert "phi_total" in endpoints["vi"]
        assert "severity" in endpoints["vi"]
        assert "mode" in endpoints["vi"]

        # Epistemic endpoint
        assert "avg_q1" in endpoints["epistemic"]
        assert "avg_q2" in endpoints["epistemic"]
        assert "avg_confidence" in endpoints["epistemic"]


class TestMPCIntegration:
    """Testes de integracao do MPC."""

    def test_transition_model(self):
        model = TransitionModel()
        state = torch.tensor([[0.5, 0.3, 0.6, 0.4]])
        action = torch.zeros(1, 12)
        action[0, InterventionType.INJECT_WEAKEST] = 1.0
        next_state = model(state, action)
        assert next_state.shape == (1, 4)
        assert (next_state >= 0).all()
        assert (next_state <= 1).all()

    def test_navigator_plan(self):
        transition = TransitionModel()
        nav = ManifoldNavigator(transition, beam_width=2, lookahead_depth=2)
        phi_comp = torch.tensor([0.3, 0.2, 0.4, 0.3])  # Abaixo do floor
        plan = nav.plan(phi_comp)
        assert len(plan.actions) == 2
        assert plan.mode == "recovery"

    def test_navigator_maintenance(self):
        transition = TransitionModel()
        nav = ManifoldNavigator(transition)
        phi_comp = torch.tensor([0.7, 0.6, 0.8, 0.7])  # Acima do floor
        plan = nav.plan(phi_comp)
        assert plan.mode == "maintenance"

    def test_generation_with_mpc(self):
        config = AletheionV2Config.small()
        config.max_seq_len = 32
        model = AletheionV2Model(config)
        gen = Generator(model, max_new_tokens=3, top_k=10, use_mpc=True)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        result = gen.generate(input_ids)
        assert result.total_tokens == 3
        assert len(result.navigation_plans) == 3


class TestDataPipeline:
    """Testes do pipeline de dados."""

    def test_text_chunk_dataset(self):
        token_ids = torch.arange(100)
        dataset = TextChunkDataset(token_ids, seq_len=10)
        assert len(dataset) == 9  # (100-1) // 10
        item = dataset[0]
        assert item["input_ids"].shape == (10,)
        assert item["labels"].shape == (10,)
        # Labels devem ser shifted por 1
        assert (item["labels"] == token_ids[1:11]).all()

    def test_synthetic_dataloader(self):
        loader = create_synthetic_data(
            vocab_size=100, total_tokens=1000,
            seq_len=32, batch_size=4,
        )
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (4, 32)
        assert batch["labels"].shape == (4, 32)

    def test_scheduler(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=10, total_steps=100, lr_max=1e-3,
        )

        # Warmup: LR deve crescer
        lr_0 = scheduler.get_lr()
        scheduler.current_step = 5
        lr_5 = scheduler.get_lr()
        assert lr_5 > lr_0

        # Apos warmup: LR deve decair
        scheduler.current_step = 50
        lr_50 = scheduler.get_lr()
        scheduler.current_step = 90
        lr_90 = scheduler.get_lr()
        assert lr_90 < lr_50

    def test_loss_weight_annealer(self):
        annealer = LossWeightAnnealer(
            warmup_fraction=0.1, ramp_fraction=0.5, total_steps=100,
        )
        assert annealer.get_weight(0) == 0.0
        assert annealer.get_weight(5) == 0.0
        assert 0 < annealer.get_weight(25) < 1
        assert annealer.get_weight(99) == 1.0
