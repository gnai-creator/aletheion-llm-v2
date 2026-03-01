"""
Testes do modulo DRM: manifold, metric tensor, directional field, geodesic.
"""

import torch
import pytest

from aletheion_v2.drm.manifold_embedding import ManifoldEmbedding, AnchorPoints
from aletheion_v2.drm.metric_tensor import LearnableMetricTensor
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
        # Truth deve ter baixa incerteza e alta qualidade
        assert truth[0].item() < 0.3  # q1 baixo
        assert truth[1].item() < 0.3  # q2 baixo
        assert truth[4].item() > 0.7  # qualidade alta

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
    """Testes do tensor metrico."""

    def setup_method(self):
        self.metric = LearnableMetricTensor(dim=5)

    def test_spd(self):
        """G deve ser simetrica positiva definida."""
        G = self.metric()
        assert G.shape == (5, 5)
        # Simetria
        assert torch.allclose(G, G.t(), atol=1e-6)
        # Positiva definida (eigenvalues > 0)
        eigenvalues = torch.linalg.eigvalsh(G)
        assert (eigenvalues > 0).all()

    def test_initial_identity_like(self):
        """Inicializacao deve ser proxima de identidade."""
        G = self.metric()
        I = torch.eye(5)
        # Nao e exatamente I por causa do softplus + eps
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


class TestDirectionalField:
    """Testes do campo direcional."""

    def setup_method(self):
        self.n_heads = 4
        self.n_layers = 2
        self.field = DirectionalField(self.n_heads, self.n_layers, drm_dim=5)

    def test_output_shapes(self):
        attn = torch.rand(2, self.n_layers, self.n_heads, 8, 8)
        # Normaliza para ser attention valida
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
        # Atencao uniforme -> entropia maxima
        T = 8
        attn = torch.ones(1, 2, T, T) / T
        entropy = extractor(attn)
        # Entropia normalizada deve ser ~1.0
        assert entropy.mean().item() > 0.9

    def test_focused_attention_low_entropy(self):
        extractor = AttentionEntropyExtractor(n_heads=2)
        T = 8
        # Atencao focada no primeiro token
        attn = torch.zeros(1, 2, T, T)
        attn[:, :, :, 0] = 1.0
        entropy = extractor(attn)
        # Entropia deve ser ~0
        assert entropy.mean().item() < 0.1


class TestGeodesicDistance:
    """Testes de distancia geodesica."""

    def setup_method(self):
        self.geo = GeodesicDistance(drm_dim=5)

    def test_zero_distance_at_centroid(self):
        truth = torch.tensor([0.1, 0.1, 0.5, 0.9, 0.9])
        coords = truth.unsqueeze(0).unsqueeze(0)  # [1, 1, 5]
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
        # G escalado deve dar distancia maior
        assert d2.item() > d1.item()

    def test_batch_to_anchors(self):
        coords = torch.rand(2, 10, 5)
        anchors = torch.rand(6, 5)
        G = torch.eye(5)
        dists = self.geo.batch_to_anchors(coords, anchors, G)
        assert dists.shape == (2, 10, 6)
        assert (dists >= 0).all()
