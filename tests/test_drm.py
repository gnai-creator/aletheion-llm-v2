"""
Testes do modulo DRM: manifold, metric tensor, MetricNet, directional field, geodesic.
"""

import torch
import pytest

from aletheion_v2.drm.manifold_embedding import ManifoldEmbedding, AnchorPoints
from aletheion_v2.drm.metric_tensor import LearnableMetricTensor, MetricNet
from aletheion_v2.drm.directional_field import (
    DirectionalField, AttentionEntropyExtractor,
)
from aletheion_v2.drm.geodesic_distance import GeodesicDistance


class TestAnchorPoints:
    """Testes dos pontos ancora."""

    def test_anchor_shapes(self):
        anchors = AnchorPoints(drm_dim=5, num_anchors=6)
        assert anchors.anchors.shape == (6, 5)

    def test_truth_centroid(self):
        anchors = AnchorPoints()
        truth = anchors.truth_centroid
        assert truth.shape == (5,)
        assert truth[0].item() < 0.3
        assert truth[1].item() < 0.3
        assert truth[4].item() > 0.7

    def test_anchors_in_range(self):
        anchors = AnchorPoints()
        assert (anchors.anchors >= 0).all()
        assert (anchors.anchors <= 1).all()


class TestManifoldEmbedding:
    """Testes do manifold embedding."""

    def setup_method(self):
        self.d_model = 64
        self.drm_dim = 5
        self.emb = ManifoldEmbedding(self.d_model, self.drm_dim)

    def test_output_shape(self):
        hidden = torch.randn(2, 10, self.d_model)
        coords, anchor_dists = self.emb(hidden)
        assert coords.shape == (2, 10, 5)
        assert anchor_dists.shape == (2, 10, 6)

    def test_coords_in_range(self):
        hidden = torch.randn(4, 8, self.d_model)
        coords, _ = self.emb(hidden)
        assert (coords >= 0).all()
        assert (coords <= 1).all()

    def test_gradient_flow(self):
        hidden = torch.randn(2, 5, self.d_model, requires_grad=True)
        coords, _ = self.emb(hidden)
        loss = coords.sum()
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.abs().sum() > 0

    def test_different_inputs_different_coords(self):
        h1 = torch.randn(1, 5, self.d_model)
        h2 = torch.randn(1, 5, self.d_model) * 10
        c1, _ = self.emb(h1)
        c2, _ = self.emb(h2)
        assert not torch.allclose(c1, c2)


class TestLearnableMetricTensor:
    """Testes do tensor metrico constante."""

    def setup_method(self):
        self.metric = LearnableMetricTensor(dim=5)

    def test_spd(self):
        G = self.metric()
        assert G.shape == (5, 5)
        assert torch.allclose(G, G.t(), atol=1e-6)
        eigenvalues = torch.linalg.eigvalsh(G)
        assert (eigenvalues > 0).all()

    def test_initial_identity_like(self):
        G = self.metric()
        assert (G.diag() > 0.5).all()

    def test_condition_number(self):
        kappa = self.metric.condition_number()
        assert kappa.item() >= 1.0

    def test_gradient_flow(self):
        G = self.metric()
        loss = G.sum()
        loss.backward()
        assert self.metric.L_params.grad is not None

    def test_inverse(self):
        G = self.metric()
        G_inv = self.metric.inverse()
        product = G @ G_inv
        I = torch.eye(5)
        assert torch.allclose(product, I, atol=1e-4)


class TestMetricNet:
    """Testes do campo tensorial G(x) com curvatura real."""

    def setup_method(self):
        self.net = MetricNet(dim=5, hidden_dim=32, n_quad=5)

    def test_output_shape(self):
        """G(x) deve ter shape [B, T, D, D]."""
        coords = torch.rand(2, 8, 5)
        G = self.net(coords)
        assert G.shape == (2, 8, 5, 5)

    def test_spd_per_point(self):
        """G(x) deve ser SPD em cada ponto."""
        coords = torch.rand(3, 6, 5)
        G = self.net(coords)
        # Flatten batch para verificar cada ponto
        G_flat = G.reshape(-1, 5, 5)
        for i in range(G_flat.shape[0]):
            Gi = G_flat[i]
            # Simetria
            assert torch.allclose(Gi, Gi.t(), atol=1e-5), f"G[{i}] nao e simetrica"
            # Positiva definida
            eigvals = torch.linalg.eigvalsh(Gi)
            assert (eigvals > 0).all(), f"G[{i}] tem eigenvalue <= 0: {eigvals}"

    def test_symmetry_per_point(self):
        """G(x) deve ser simetrica em cada ponto."""
        coords = torch.rand(2, 4, 5)
        G = self.net(coords)
        assert torch.allclose(G, G.transpose(-1, -2), atol=1e-6)

    def test_identity_init(self):
        """Com zero init, G(x) deve ser proximo de c*I."""
        coords = torch.rand(1, 1, 5)
        G = self.net(coords)
        G_single = G[0, 0]
        # Diagonal deve ser dominante (proximo de identidade escalada)
        diag = G_single.diag()
        off_diag_norm = (G_single - torch.diag(diag)).norm()
        diag_norm = diag.norm()
        # Off-diagonal deve ser pequeno relativo a diagonal
        assert off_diag_norm / diag_norm < 0.1, (
            f"Off-diagonal muito grande: {off_diag_norm:.4f} vs diag {diag_norm:.4f}"
        )

    def test_gradient_flow_params(self):
        """Gradientes devem fluir para params da rede."""
        coords = torch.rand(2, 4, 5)
        G = self.net(coords)
        loss = G.sum()
        loss.backward()
        for p in self.net.net.parameters():
            assert p.grad is not None

    def test_gradient_flow_coords(self):
        """Gradientes devem fluir para coords (apos quebrar simetria do zero init)."""
        # Primeiro, treina um step para que a rede nao seja constante
        coords_train = torch.rand(2, 4, 5)
        G = self.net(coords_train)
        loss = (G - 2.0 * torch.eye(5)).pow(2).sum()
        loss.backward()
        opt = torch.optim.SGD(self.net.parameters(), lr=0.1)
        opt.step()
        opt.zero_grad()

        # Agora gradiente deve fluir para coords
        coords = torch.rand(2, 4, 5, requires_grad=True)
        G = self.net(coords)
        loss = G.sum()
        loss.backward()
        assert coords.grad is not None
        assert coords.grad.abs().sum() > 0

    def test_varies_with_position(self):
        """G(x) deve variar com a posicao (apos treino).

        Com zero init, G e quase constante, mas nao exatamente
        porque softplus(0) + rede nao e perfeitamente constante.
        """
        # Treina um step para quebrar simetria
        coords = torch.rand(2, 4, 5, requires_grad=True)
        G = self.net(coords)
        target = torch.eye(5).expand_as(G) * 2.0
        loss = (G - target).pow(2).sum()
        loss.backward()

        # Apos backward, verifica que coords diferentes dao G diferentes
        opt = torch.optim.SGD(self.net.parameters(), lr=0.1)
        opt.step()

        c1 = torch.tensor([[[0.1, 0.1, 0.1, 0.1, 0.1]]])
        c2 = torch.tensor([[[0.9, 0.9, 0.9, 0.9, 0.9]]])
        G1 = self.net(c1)
        G2 = self.net(c2)
        assert not torch.allclose(G1, G2, atol=1e-4), "G deve variar com posicao"

    def test_line_integral_distance_shape(self):
        """Integral de linha deve retornar [B, T, 1]."""
        p = torch.rand(2, 8, 5)
        q = torch.rand(5)  # truth centroid
        dist = self.net.line_integral_distance(p, q)
        assert dist.shape == (2, 8, 1)

    def test_line_integral_distance_positive(self):
        """Distancia deve ser positiva para pontos distintos."""
        p = torch.rand(2, 4, 5) * 0.3
        q = torch.tensor([0.9, 0.9, 0.5, 0.5, 0.5])
        dist = self.net.line_integral_distance(p, q)
        assert (dist > 0).all()

    def test_line_integral_distance_zero_at_same_point(self):
        """Distancia deve ser ~0 quando p == q."""
        q = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        p = q.unsqueeze(0).unsqueeze(0)  # [1, 1, 5]
        dist = self.net.line_integral_distance(p, q)
        assert dist.item() < 1e-3

    def test_line_integral_gradient_flow(self):
        """Gradientes devem fluir pela integral de linha."""
        p = torch.rand(2, 4, 5, requires_grad=True)
        q = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        dist = self.net.line_integral_distance(p, q)
        loss = dist.sum()
        loss.backward()
        assert p.grad is not None
        assert p.grad.abs().sum() > 0

    def test_forward_constant(self):
        """forward_constant deve retornar [D, D]."""
        G = self.net.forward_constant()
        assert G.shape == (5, 5)


class TestDirectionalField:
    """Testes do campo direcional."""

    def setup_method(self):
        self.n_heads = 4
        self.n_layers = 2
        self.field = DirectionalField(self.n_heads, self.n_layers, drm_dim=5)

    def test_output_shapes(self):
        attn = torch.rand(2, self.n_layers, self.n_heads, 8, 8)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        directions, dim_D = self.field(attn)
        assert directions.shape == (2, 8, 5)
        assert dim_D.shape == (2, 8, 1)

    def test_dim_D_range(self):
        attn = torch.rand(2, self.n_layers, self.n_heads, 8, 8)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        _, dim_D = self.field(attn)
        assert (dim_D >= 1.0).all()
        assert (dim_D <= 5.0).all()

    def test_gradient_flow(self):
        attn = torch.rand(2, self.n_layers, self.n_heads, 8, 8, requires_grad=True)
        directions, dim_D = self.field(attn)
        loss = directions.sum() + dim_D.sum()
        loss.backward()
        assert attn.grad is not None


class TestAttentionEntropyExtractor:
    """Testes do extrator de entropia."""

    def test_uniform_attention_max_entropy(self):
        extractor = AttentionEntropyExtractor(n_heads=2)
        T = 8
        attn = torch.ones(1, 2, T, T) / T
        entropy = extractor(attn)
        assert entropy.mean().item() > 0.9

    def test_focused_attention_low_entropy(self):
        extractor = AttentionEntropyExtractor(n_heads=2)
        T = 8
        attn = torch.zeros(1, 2, T, T)
        attn[:, :, :, 0] = 1.0
        entropy = extractor(attn)
        assert entropy.mean().item() < 0.1


class TestGeodesicDistance:
    """Testes de distancia geodesica."""

    def setup_method(self):
        self.geo = GeodesicDistance(drm_dim=5)

    def test_zero_distance_at_centroid(self):
        truth = torch.tensor([0.1, 0.1, 0.5, 0.9, 0.9])
        coords = truth.unsqueeze(0).unsqueeze(0)
        G = torch.eye(5)
        dist = self.geo(coords, truth, G)
        assert dist.item() < 1e-3

    def test_positive_distance(self):
        truth = torch.tensor([0.1, 0.1, 0.5, 0.9, 0.9])
        coords = torch.tensor([[[0.9, 0.9, 0.5, 0.1, 0.1]]])
        G = torch.eye(5)
        dist = self.geo(coords, truth, G)
        assert dist.item() > 0.5

    def test_metric_affects_distance(self):
        truth = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        coords = torch.tensor([[[0.6, 0.5, 0.5, 0.5, 0.5]]])
        G_id = torch.eye(5)
        G_scaled = torch.eye(5) * 4.0
        d1 = self.geo(coords, truth, G_id)
        d2 = self.geo(coords, truth, G_scaled)
        assert d2.item() > d1.item()

    def test_batch_to_anchors(self):
        coords = torch.rand(2, 10, 5)
        anchors = torch.rand(6, 5)
        G = torch.eye(5)
        dists = self.geo.batch_to_anchors(coords, anchors, G)
        assert dists.shape == (2, 10, 6)
        assert (dists >= 0).all()

    def test_with_metric_net(self):
        """Distancia com MetricNet deve funcionar e retornar shape correto."""
        net = MetricNet(dim=5, hidden_dim=16, n_quad=3)
        truth = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        coords = torch.rand(2, 4, 5)
        G = torch.eye(5)  # fallback
        dist = self.geo(coords, truth, G, metric_net=net)
        assert dist.shape == (2, 4, 1)
        assert (dist > 0).all()

    def test_batch_to_anchors_with_metric_net(self):
        """batch_to_anchors com MetricNet."""
        net = MetricNet(dim=5, hidden_dim=16, n_quad=3)
        coords = torch.rand(2, 4, 5)
        anchors = torch.rand(6, 5)
        G = torch.eye(5)
        dists = self.geo.batch_to_anchors(coords, anchors, G, metric_net=net)
        assert dists.shape == (2, 4, 6)
        assert (dists >= 0).all()
