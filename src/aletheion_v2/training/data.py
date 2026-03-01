"""
Dataset e DataLoader para treinamento do AletheionV2.

Suporta:
- Texto bruto (lista de strings tokenizadas)
- Formato HuggingFace datasets (WikiText-2, etc)
- Chunking automatico para sequencias de comprimento fixo
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Iterator
import math


class TextChunkDataset(Dataset):
    """Dataset de chunks de texto com comprimento fixo.

    Recebe um tensor 1D de token ids e divide em chunks
    de seq_len tokens. O label e o token seguinte (shift right).

    Args:
        token_ids: Tensor 1D com todos os token ids concatenados
        seq_len: Comprimento de cada chunk
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len
        # Numero de chunks completos (precisa seq_len + 1 por label)
        self.n_chunks = (len(token_ids) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict:
        """Retorna input_ids e labels para chunk idx.

        Returns:
            dict com 'input_ids' [seq_len] e 'labels' [seq_len]
        """
        start = idx * self.seq_len
        end = start + self.seq_len

        input_ids = self.token_ids[start:end]
        labels = self.token_ids[start + 1 : end + 1]

        return {"input_ids": input_ids, "labels": labels}


def tokenize_text_simple(texts: List[str], vocab_size: int = 32000) -> torch.Tensor:
    """Tokenizacao simples (byte-level) para testes.

    Cada byte vira um token id. Para producao, usar tiktoken.

    Args:
        texts: Lista de textos
        vocab_size: Tamanho do vocabulario (para clamp)

    Returns:
        token_ids: Tensor 1D
    """
    all_ids = []
    for text in texts:
        ids = [min(b, vocab_size - 1) for b in text.encode("utf-8")]
        all_ids.extend(ids)
    return torch.tensor(all_ids, dtype=torch.long)


def create_dataloader(
    token_ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Cria DataLoader a partir de token ids.

    Args:
        token_ids: Tensor 1D de token ids
        seq_len: Comprimento de sequencia
        batch_size: Tamanho do batch
        shuffle: Se True, embaralha chunks
        num_workers: Workers para loading paralelo

    Returns:
        DataLoader configurado
    """
    dataset = TextChunkDataset(token_ids, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def create_synthetic_data(
    vocab_size: int = 32000,
    total_tokens: int = 100_000,
    seq_len: int = 256,
    batch_size: int = 8,
) -> DataLoader:
    """Cria dados sinteticos para teste rapido.

    Gera token ids aleatorios. Util para validar que o modelo
    roda sem erros antes de usar dados reais.

    Returns:
        DataLoader com dados sinteticos
    """
    token_ids = torch.randint(0, vocab_size, (total_tokens,))
    return create_dataloader(token_ids, seq_len, batch_size, shuffle=True)
