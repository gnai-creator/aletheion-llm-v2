"""
Transformer Block com retorno de attention patterns.

Implementa Multi-Head Self-Attention com RoPE e FFN com GeLU.
Retorna attention_weights para uso no DirectionalField.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from aletheion_v2.core.embeddings import RotaryEmbedding, apply_rotary_emb


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention com RoPE.

    Retorna attention_weights para analise de direcionalidade.
    Suporta mascara causal e KV-cache (futuro).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve ser divisivel por n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_base)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: [B, T, d_model]
            mask: [T, T] mascara causal (opcional)
            return_weights: se True, retorna attention weights

        Returns:
            output: [B, T, d_model]
            attn_weights: [B, n_heads, T, T] ou None
        """
        B, T, _ = x.shape

        # Projecoes Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # [B, n_heads, T, head_dim]

        # Aplica RoPE em Q e K
        cos, sin = self.rope(T)
        cos = cos.to(q.device, dtype=q.dtype)
        sin = sin.to(q.device, dtype=q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, T, T]

        # Mascara causal
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # Attention output
        out = torch.matmul(attn_weights_dropped, v)  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)

        if return_weights:
            return out, attn_weights
        return out, None


class FeedForward(nn.Module):
    """Feed-Forward Network com GeLU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    """Bloco Transformer com Pre-LN e retorno de attention patterns.

    Arquitetura:
        x -> LN -> MHSA -> + -> LN -> FFN -> +
        |__________________|   |______________|
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model, n_heads, max_seq_len, dropout, rope_base
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: [B, T, d_model]
            mask: [T, T] mascara causal

        Returns:
            output: [B, T, d_model]
            attn_weights: [B, n_heads, T, T] ou None
        """
        # Self-attention com residual
        normed = self.ln1(x)
        attn_out, attn_weights = self.attn(normed, mask, return_weights)
        x = x + self.dropout(attn_out)

        # FFN com residual
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x, attn_weights
