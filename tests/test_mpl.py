"""Testes do modulo MPL: DensityTracker e FrontierHead."""

import torch
import pytest

from aletheion_v2.mpl.density_tracker import DensityTracker, DensityInfo
from aletheion_v2.mpl.frontier_head import FrontierHead, FrontierOutput
from aletheion_v2.loss.frontier_loss import FrontierRegularization


class TestDensityTracker:
    """Testes do DensityTracker."""

    def setup_method(self):
        self.tracker = DensityTracker(resolution=10, bandwidth=0.15)

    def test_initial_empty(self):
        assert self.tracker.num_cells == 0
        assert self.tracker._total_updates == 0

    def test_update_adds_cells(self):
        coords = torch.rand(2, 8, 5)
        self.tracker.update(coords)
        assert self.tracker.num_cells > 0
        assert self.tracker._total_updates == 16

    def test_query_output_type(self):
        coords = torch.rand(2, 8, 5)
        self.tracker.update(coords)
        info = self.tracker.query(coords)
        assert isinstance(info, DensityInfo)

    def test_query_output_shapes(self):
        coords = torch.rand(2, 8, 5)
        self.tracker.update(coords)
        info = self.tracker.query(coords)
        assert info.density.shape == (2, 8, 1)
        assert info.novelty.shape == (2, 8, 1)

    def test_density_range(self):
        coords = torch.rand(2, 8, 5)
        self.tracker.update(coords)
        info = self.tracker.query(coords)
        assert (info.density >= 0).all()
        assert (info.density <= 1).all()

    def test_novelty_complement(self):
        coords = torch.rand(2, 8, 5)
        self.tracker.update(coords)
        info = self.tracker.query(coords)
        total = info.density + info.novelty
        assert torch.allclose(total, torch.ones_like(total))

    def test_repeated_coords_high_density(self):
        """Coordenadas repetidas devem ter alta densidade."""
        coords = torch.ones(4, 8, 5) * 0.5
        for _ in range(10):
            self.tracker.update(coords)
        info = self.tracker.query(coords)
        assert info.density.mean().item() > 0.5

    def test_empty_query_zero_density(self):
        """Sem updates, densidade deve ser zero."""
        coords = torch.rand(2, 8, 5)
        info = self.tracker.query(coords)
        assert (info.density == 0).all()

    def test_reset(self):
        coords = torch.rand(2, 8, 5)
        self.tracker.update(coords)
        self.tracker.reset()
        assert self.tracker.num_cells == 0


class TestFrontierHead:
    """Testes do FrontierHead."""

    def setup_method(self):
        self.head = FrontierHead(drm_dim=5, hidden_dim=16)
        self.B = 2
        self.T = 8

    def test_output_type(self):
        coords = torch.rand(self.B, self.T, 5)
        density = torch.rand(self.B, self.T, 1)
        out = self.head(coords, density)
        assert isinstance(out, FrontierOutput)

    def test_output_shapes(self):
        coords = torch.rand(self.B, self.T, 5)
        density = torch.rand(self.B, self.T, 1)
        out = self.head(coords, density)
        assert out.frontier_score.shape == (self.B, self.T, 1)
        assert out.novelty_enhanced.shape == (self.B, self.T, 1)

    def test_frontier_range(self):
        coords = torch.rand(self.B, self.T, 5)
        density = torch.rand(self.B, self.T, 1)
        out = self.head(coords, density)
        assert (out.frontier_score >= 0).all()
        assert (out.frontier_score <= 1).all()

    def test_high_density_low_frontier(self):
        """Alta densidade -> baixa novidade -> frontier baixo."""
        coords = torch.rand(self.B, self.T, 5)
        high_density = torch.ones(self.B, self.T, 1) * 0.99
        out = self.head(coords, high_density)
        # novelty = 0.01, score multiplicado por novidade
        assert out.frontier_score.mean().item() < 0.1

    def test_gradient_flow(self):
        coords = torch.rand(self.B, self.T, 5, requires_grad=True)
        density = torch.rand(self.B, self.T, 1)
        out = self.head(coords, density)
        loss = out.frontier_score.sum()
        loss.backward()
        assert coords.grad is not None

    def test_param_count(self):
        n_params = sum(p.numel() for p in self.head.parameters())
        # scorer: 6*16+16+16*1+1 = 129
        assert 100 <= n_params <= 150


class TestFrontierRegularization:
    """Testes da loss de fronteira."""

    def setup_method(self):
        self.loss_fn = FrontierRegularization()

    def test_high_frontier_low_loss(self):
        """Alta fronteira + alta novidade -> loss baixa."""
        frontier = torch.ones(2, 8, 1) * 0.9
        novelty = torch.ones(2, 8, 1) * 0.9
        loss = self.loss_fn(frontier, novelty)
        # -0.9 * log(0.9+eps) = -0.9*(-0.105) = 0.095 (positivo mas baixo)
        assert loss.item() < 0.5

    def test_zero_frontier_zero_loss(self):
        frontier = torch.zeros(2, 8, 1)
        novelty = torch.ones(2, 8, 1)
        loss = self.loss_fn(frontier, novelty)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flow(self):
        frontier = torch.rand(2, 8, 1, requires_grad=True)
        novelty = torch.rand(2, 8, 1)
        loss = self.loss_fn(frontier, novelty)
        loss.backward()
        assert frontier.grad is not None
