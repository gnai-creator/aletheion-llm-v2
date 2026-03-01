"""
Data Pipeline escalavel para treinamento de LLMs.

Suporta:
- Dados pre-tokenizados (binario numpy memmap)
- HuggingFace datasets com streaming
- Sharding para treinamento distribuido
- Data mixing com pesos configuraveis
- Chunking automatico para sequencias de comprimento fixo

Para treinamento em escala (7B+), usa-se o fluxo:
    1. prepare_data.py tokeniza e salva em .bin
    2. MemmapDataset le os .bin com memmap (zero RAM overhead)
    3. DataLoader com sharding por rank (DDP)
"""

import os
import math
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Optional, Dict, Tuple, Iterator
from pathlib import Path


class MemmapDataset(Dataset):
    """Dataset que le tokens pre-tokenizados via numpy memmap.

    Zero overhead de RAM - o SO faz page-in sob demanda.
    Ideal para datasets grandes (100GB+).

    Args:
        bin_path: Caminho do arquivo .bin de tokens
        seq_len: Comprimento de cada sequencia
        dtype: Tipo do numpy array ("uint16" ou "uint32")
    """

    def __init__(
        self,
        bin_path: str,
        seq_len: int,
        dtype: str = "uint16",
    ):
        self.seq_len = seq_len
        np_dtype = np.uint16 if dtype == "uint16" else np.uint32
        self.data = np.memmap(bin_path, dtype=np_dtype, mode="r")
        self.n_chunks = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        input_ids = torch.from_numpy(self.data[start:end].astype(np.int64))
        labels = torch.from_numpy(self.data[start + 1 : end + 1].astype(np.int64))
        return {"input_ids": input_ids, "labels": labels}


class ShardedMemmapDataset(Dataset):
    """Dataset que le de multiplos shards .bin.

    Distribui shards entre ranks no treinamento distribuido.
    Cada rank processa um subconjunto dos shards.

    Args:
        shard_dir: Diretorio contendo arquivos .bin
        seq_len: Comprimento de cada sequencia
        rank: Rank do processo atual (DDP)
        world_size: Numero total de processos
        dtype: Tipo do numpy array
    """

    def __init__(
        self,
        shard_dir: str,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        dtype: str = "uint16",
    ):
        self.seq_len = seq_len
        self.dtype = dtype
        np_dtype = np.uint16 if dtype == "uint16" else np.uint32

        # Descobre todos os shards
        shard_paths = sorted(Path(shard_dir).glob("*.bin"))
        if not shard_paths:
            raise FileNotFoundError(f"Nenhum shard .bin encontrado em {shard_dir}")

        # Distribui shards por rank
        self.my_shards = [
            str(p) for i, p in enumerate(shard_paths) if i % world_size == rank
        ]

        # Carrega memmaps de cada shard e calcula offsets
        self.shards = []
        self.cumulative_chunks = [0]
        total_chunks = 0

        for path in self.my_shards:
            mmap = np.memmap(path, dtype=np_dtype, mode="r")
            n_chunks = (len(mmap) - 1) // seq_len
            self.shards.append(mmap)
            total_chunks += n_chunks
            self.cumulative_chunks.append(total_chunks)

        self.total_chunks = total_chunks

    def __len__(self) -> int:
        return self.total_chunks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Encontra shard e offset local
        shard_idx = 0
        for i in range(len(self.cumulative_chunks) - 1):
            if idx < self.cumulative_chunks[i + 1]:
                shard_idx = i
                break

        local_idx = idx - self.cumulative_chunks[shard_idx]
        data = self.shards[shard_idx]

        start = local_idx * self.seq_len
        end = start + self.seq_len

        input_ids = torch.from_numpy(data[start:end].astype(np.int64))
        labels = torch.from_numpy(data[start + 1 : end + 1].astype(np.int64))
        return {"input_ids": input_ids, "labels": labels}


class StreamingHFDataset(IterableDataset):
    """Dataset streaming de HuggingFace datasets.

    Nao baixa tudo de uma vez - faz streaming do hub.
    Tokeniza on-the-fly e chunka para seq_len.

    Args:
        dataset_name: Nome HuggingFace (ex: "HuggingFaceFW/fineweb")
        subset: Subset (ex: "sample-10BT")
        tokenizer: AletheionTokenizer
        seq_len: Comprimento da sequencia
        split: Split do dataset ("train", "validation")
        text_field: Campo de texto no dataset
        rank: Rank DDP
        world_size: World size DDP
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        seq_len: int,
        subset: str = "",
        split: str = "train",
        text_field: str = "text",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.text_field = text_field
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets nao instalado. Instale com: pip install datasets"
            )

        # Carrega em streaming
        kwargs = {"streaming": True, "split": self.split}
        if self.subset:
            kwargs["name"] = self.subset

        ds = load_dataset(self.dataset_name, **kwargs)

        # Buffer de tokens para chunking
        buffer = []

        for i, example in enumerate(ds):
            # Sharding por rank
            if i % self.world_size != self.rank:
                continue

            text = example.get(self.text_field, "")
            if not text:
                continue

            tokens = self.tokenizer.encode(text, add_eos=True)
            buffer.extend(tokens)

            # Emite chunks completos
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


class MixedDataset(Dataset):
    """Dataset que mixa multiplas fontes com pesos.

    Exemplo: 70% web + 20% books + 10% code.

    Args:
        datasets: Lista de datasets
        weights: Pesos de cada dataset (somam 1.0)
        total_samples: Numero total de amostras por epoch
    """

    def __init__(
        self,
        datasets: List[Dataset],
        weights: List[float],
        total_samples: int = 100_000,
    ):
        assert len(datasets) == len(weights)
        self.datasets = datasets
        self.total_samples = total_samples

        # Normaliza pesos
        total_w = sum(weights)
        self.weights = [w / total_w for w in weights]

        # Pre-computa quantas amostras de cada dataset
        self.samples_per_dataset = [
            int(total_samples * w) for w in self.weights
        ]
        # Ajusta ultimo para totalizar exato
        diff = total_samples - sum(self.samples_per_dataset)
        self.samples_per_dataset[-1] += diff

        # Gera indices
        self._build_indices()

    def _build_indices(self) -> None:
        """Pre-gera indices aleatorios para cada dataset."""
        self.indices = []
        rng = np.random.default_rng(42)

        for ds_idx, (ds, n_samples) in enumerate(
            zip(self.datasets, self.samples_per_dataset)
        ):
            if len(ds) == 0:
                continue
            local_indices = rng.integers(0, len(ds), size=n_samples)
            for local_idx in local_indices:
                self.indices.append((ds_idx, int(local_idx)))

        rng.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx, local_idx = self.indices[idx]
        return self.datasets[ds_idx][local_idx]


def create_dataloader_from_config(
    config,
    tokenizer=None,
    split: str = "train",
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Factory: cria DataLoader a partir da config.

    Detecta automaticamente o tipo de dados:
    1. Se data_dir tem .bin -> MemmapDataset / ShardedMemmapDataset
    2. Se dataset_name definido -> StreamingHFDataset
    3. Senao -> dados sinteticos

    Args:
        config: AletheionV2Config
        tokenizer: AletheionTokenizer (necessario para streaming)
        split: "train" ou "validation"
        rank: Rank DDP
        world_size: World size DDP

    Returns:
        DataLoader configurado
    """
    data_dir = Path(config.data_dir)
    seq_len = config.max_seq_len

    # 1. Dados pre-tokenizados
    if data_dir.exists():
        bin_files = list(data_dir.glob(f"{split}*.bin"))
        if not bin_files:
            bin_files = list(data_dir.glob("*.bin"))

        if bin_files:
            if len(bin_files) == 1:
                dataset = MemmapDataset(str(bin_files[0]), seq_len)
            else:
                dataset = ShardedMemmapDataset(
                    str(data_dir), seq_len, rank, world_size
                )

            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=(split == "train"),
                num_workers=config.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )

    # 2. Streaming HuggingFace
    if config.dataset_name:
        if tokenizer is None:
            from aletheion_v2.tokenizer import AletheionTokenizer
            tokenizer = AletheionTokenizer(config.tokenizer_name)

        dataset = StreamingHFDataset(
            config.dataset_name,
            tokenizer,
            seq_len,
            subset=config.dataset_subset,
            split=split,
            rank=rank,
            world_size=world_size,
        )

        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # 3. Fallback: dados sinteticos
    from aletheion_v2.training.data import create_synthetic_data
    return create_synthetic_data(
        vocab_size=config.vocab_size,
        seq_len=seq_len,
        batch_size=config.batch_size,
    )
