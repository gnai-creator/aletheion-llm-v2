"""
Embeddings: Token embedding + Rotary Position Encoding (RoPE).

RoPE aplica rotacoes no espaco complexo para codificar posicao
sem parametros adicionais. Compativel com KV-cache.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Gera frequencias de rotacao para cada par de dimensoes.
    Aplica rotacao no espaco complexo aos vetores Q e K.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Frequencias inversas: theta_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-computa cos/sin para posicoes
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Pre-computa cos e sin para todas posicoes."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [T, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (cos, sin) para seq_len posicoes.

        Returns:
            cos: [seq_len, dim]
            sin: [seq_len, dim]
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Aplica RoPE a tensor x.

    Args:
        x: [B, n_heads, T, head_dim]
        cos: [T, head_dim]
        sin: [T, head_dim]

    Returns:
        x_rotated: [B, n_heads, T, head_dim]
    """
    # Reshape para broadcast: [1, 1, T, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Rotacao: separa pares de dimensoes
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Aplica rotacao no espaco complexo
    rotated = torch.cat([
        x1 * cos[..., : cos.shape[-1] // 2] - x2 * sin[..., : sin.shape[-1] // 2],
        x2 * cos[..., cos.shape[-1] // 2 :] + x1 * sin[..., sin.shape[-1] // 2 :],
    ], dim=-1)

    return rotated


class TokenEmbedding(nn.Module):
    """Embedding de tokens com RoPE.

    Combina token embedding lookup com RoPE para posicao.
    Inclui dropout e layer norm pre-transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # RoPE e inicializado no head_dim, mas guardamos aqui para acesso
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len

        # Inicializacao dos pesos
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Converte input_ids em embeddings.

        Args:
            input_ids: [B, T] token ids

        Returns:
            embeddings: [B, T, d_model]
        """
        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        return self.dropout(x)
