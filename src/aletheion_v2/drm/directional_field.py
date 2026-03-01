"""
Directional Field: Extrai campo direcional D(p) dos attention patterns.

Converte attention weights em dimensionalidade direcional suave (dim_D).
Usa entropia dos padroes de atencao como proxy para riqueza direcional.

Formulas:
    H(attn) = -sum(attn * log(attn))           -- entropia por head
    dim_D_raw = proj(H_concat)                  -- projecao aprendida
    dim_D = sigmoid(dim_D_raw) * (d_max - 1) + 1  -- dim em [1, d_max]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class AttentionEntropyExtractor(nn.Module):
    """Extrai entropia dos attention patterns por head.

    A entropia mede a "espalhamento" da atencao:
    - Entropia alta = atencao distribuida = muitas direcoes ativas
    - Entropia baixa = atencao focada = poucas direcoes

    Args:
        n_heads: Numero de heads de atencao
        eps: Epsilon para log numerico
    """

    def __init__(self, n_heads: int, eps: float = 1e-8):
        super().__init__()
        self.n_heads = n_heads
        self.eps = eps

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Computa entropia por head por token.

        Args:
            attn_weights: [B, n_heads, T, T] attention weights (ja softmax)

        Returns:
            entropy: [B, T, n_heads] entropia por head por posicao
        """
        # Entropia de Shannon: H = -sum(p * log(p))
        attn_clamped = attn_weights.clamp(min=self.eps)
        entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)
        # entropy: [B, n_heads, T]

        # Transpoe para [B, T, n_heads]
        entropy = entropy.transpose(1, 2)

        # Normaliza pela entropia maxima (log T)
        T = attn_weights.shape[-1]
        max_entropy = math.log(max(T, 2))
        entropy = entropy / max_entropy

        return entropy


class DirectionalProjection(nn.Module):
    """Projeta entropias de atencao para direcoes no manifold 5D.

    Aprende a mapear padroes de entropia multi-head para
    vetores direcionais no espaco epistemico.

    Args:
        n_heads: Numero de heads
        drm_dim: Dimensao do manifold
        hidden_dim: Dimensao oculta
    """

    def __init__(self, n_heads: int, drm_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_heads, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, drm_dim),
        )

    def forward(self, entropy: torch.Tensor) -> torch.Tensor:
        """Projeta entropia para direcoes.

        Args:
            entropy: [B, T, n_heads]

        Returns:
            directions: [B, T, drm_dim] vetores direcionais (nao normalizados)
        """
        return self.proj(entropy)


class DirectionalField(nn.Module):
    """Campo direcional completo: attn_patterns -> dim_D + direcoes.

    Combina:
    1. Extracao de entropia multi-head
    2. Projecao para direcoes no manifold
    3. Calculo de dim_D suave via SVD das direcoes

    Args:
        n_heads: Numero de heads de atencao
        n_layers: Numero de layers (para agregar)
        drm_dim: Dimensao do manifold
        hidden_dim: Dimensao oculta da projecao
    """

    def __init__(
        self,
        n_heads: int,
        n_layers: int,
        drm_dim: int = 5,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.drm_dim = drm_dim

        # Extrator de entropia
        self.entropy_extractor = AttentionEntropyExtractor(n_heads)

        # Projecao por layer -> combinada
        self.layer_proj = DirectionalProjection(n_heads, drm_dim, hidden_dim)

        # Agregacao multi-layer: peso aprendido por layer
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)

        # Projecao final para dim_D escalar
        self.dim_proj = nn.Sequential(
            nn.Linear(drm_dim, drm_dim),
            nn.GELU(),
            nn.Linear(drm_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        attn_patterns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computa campo direcional e dim_D.

        Args:
            attn_patterns: [B, n_layers, n_heads, T, T]

        Returns:
            directions: [B, T, drm_dim] vetores direcionais
            dim_D: [B, T, 1] dimensionalidade direcional em [1, drm_dim]
        """
        B, L, H, T, _ = attn_patterns.shape

        # Entropia por layer
        # [B, L, H, T, T] -> processa cada layer
        directions_list = []
        for layer_idx in range(L):
            attn_layer = attn_patterns[:, layer_idx]  # [B, H, T, T]
            entropy = self.entropy_extractor(attn_layer)  # [B, T, H]
            dirs = self.layer_proj(entropy)  # [B, T, drm_dim]
            directions_list.append(dirs)

        # Stack e agrega com pesos aprendiveis
        directions_stack = torch.stack(directions_list, dim=1)  # [B, L, T, drm_dim]
        weights = F.softmax(self.layer_weights, dim=0)  # [L]
        weights = weights.view(1, L, 1, 1)  # [1, L, 1, 1]
        directions = (directions_stack * weights).sum(dim=1)  # [B, T, drm_dim]

        # dim_D: dimensionalidade suave
        # Mede "riqueza" do vetor direcional
        dim_D_raw = self.dim_proj(directions)  # [B, T, 1]
        # Escala para [1, drm_dim]
        dim_D = dim_D_raw * (self.drm_dim - 1) + 1.0

        return directions, dim_D
