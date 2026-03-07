"""
AletheionV2Model: Modelo principal com tomografia epistemica.

Arquitetura:
    input_ids -> Embeddings -> TransformerBlock x N -> ln_final
                                                       |
                                            +----------+----------+
                                            |                     |
                                        lm_head              EpistemicHead
                                            |                     |
                                        logits            EpistemicTomography
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.embeddings import TokenEmbedding
from aletheion_v2.core.transformer_block import TransformerBlock
from aletheion_v2.core.output import ModelOutput, EpistemicTomography
from aletheion_v2.epistemic.epistemic_head import EpistemicHead


class AletheionV2Model(nn.Module):
    """Modelo AletheionV2 com tomografia epistemica integrada.

    Transformer decoder-only com EpistemicHead que produz
    coordenadas 5D, Q1/Q2, confianca MAD e phi(M) por token.

    Args:
        config: AletheionV2Config com todos os hiperparametros
    """

    def __init__(self, config: AletheionV2Config):
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = TokenEmbedding(
            config.vocab_size, config.d_model,
            config.max_seq_len, config.dropout, config.rope_base,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.max_seq_len, config.dropout, config.rope_base,
            )
            for _ in range(config.n_layers)
        ])

        # Layer norm final
        self.ln_final = nn.LayerNorm(config.d_model)

        # LM head (projecao para vocabulario)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Epistemic head
        self.epistemic_head = EpistemicHead(config)

        # Tie weights: lm_head.weight = embeddings.token_emb.weight
        self.lm_head.weight = self.embeddings.token_emb.weight

        # Mascara causal (pre-computada)
        self._causal_mask: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializacao dos pesos do modelo."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Gera mascara causal [T, T].

        Returns:
            mask: [T, T] com -inf acima da diagonal
        """
        if self._causal_mask is None or self._causal_mask.shape[0] < T:
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=device), diagonal=1
            )
            self._causal_mask = mask
        return self._causal_mask[:T, :T].to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_tomography: bool = True,
        state_vector: Optional[torch.Tensor] = None,
        dream_mode: bool = False,
    ) -> ModelOutput:
        """Forward pass completo.

        Args:
            input_ids: [B, T] token ids
            return_tomography: se True, computa EpistemicTomography
            state_vector: [B, 4] estado causal opcional (CausalState)
            dream_mode: se True, amplifica eidos decay

        Returns:
            ModelOutput com logits, tomography, hidden_states, attention_patterns
        """
        B, T = input_ids.shape

        # Embeddings
        x = self.embeddings(input_ids)  # [B, T, d_model]

        # Mascara causal
        mask = self._get_causal_mask(T, x.device)

        # Transformer blocks (coleta attention patterns)
        all_attn_weights = []
        need_attn = return_tomography

        for block in self.blocks:
            x, attn_w = block(x, mask, return_weights=need_attn)
            if attn_w is not None:
                # Detach: attention weights nao precisam de gradiente
                # para o EpistemicHead (ele tem seus proprios parametros).
                # Isso economiza ~3GB de VRAM no backward.
                all_attn_weights.append(attn_w.detach())

        # Layer norm final
        hidden_states = self.ln_final(x)  # [B, T, d_model]

        # LM head
        logits = self.lm_head(hidden_states)  # [B, T, V]

        # Epistemic head
        tomography = None
        attention_patterns = None

        if return_tomography and all_attn_weights:
            attention_patterns = torch.stack(all_attn_weights, dim=1)
            # [B, n_layers, n_heads, T, T]
            tomography = self.epistemic_head(
                hidden_states, attention_patterns,
                state_vector=state_vector,
                dream_mode=dream_mode,
            )

        return ModelOutput(
            logits=logits,
            tomography=tomography,
            hidden_states=hidden_states,
            attention_patterns=attention_patterns,
        )

    def count_parameters(self) -> dict:
        """Conta parametros por componente.

        Returns:
            Dict com contagem por modulo
        """
        counts = {}
        counts["embeddings"] = sum(
            p.numel() for p in self.embeddings.parameters()
        )
        counts["transformer"] = sum(
            p.numel() for p in self.blocks.parameters()
        )
        counts["ln_final"] = sum(
            p.numel() for p in self.ln_final.parameters()
        )
        # lm_head tied com embeddings, nao conta
        counts["epistemic_head"] = sum(
            p.numel() for p in self.epistemic_head.parameters()
        )
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["epistemic_pct"] = (
            counts["epistemic_head"] / counts["total"] * 100
        )
        return counts

    @torch.no_grad()
    def generate_next_token(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> tuple[torch.Tensor, Optional[EpistemicTomography]]:
        """Gera proximo token com tomografia.

        Args:
            input_ids: [B, T]
            temperature: temperatura de sampling
            top_k: top-k filtering

        Returns:
            next_token: [B, 1]
            tomography: EpistemicTomography do ultimo token
        """
        output = self.forward(input_ids, return_tomography=True)
        logits = output.logits[:, -1, :]  # [B, V]

        # Temperatura adaptativa do epistemic head (se disponivel)
        if output.tomography is not None:
            ep_temp = output.tomography.temperature[:, -1, 0].mean()
            temperature = temperature * ep_temp.item()

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            vals, _ = logits.topk(top_k)
            threshold = vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Sampling
        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

        return next_token, output.tomography
