"""
Experience Replay Buffer para treinamento continuo.

Implementa reservoir sampling (Vitter, 1985) para manter uma amostra
uniforme de batches de fases anteriores. Mistura dados antigos com
novos durante treinamento para mitigar catastrophic forgetting.

Baseado no survey Wang et al. (2302.00487):
- Experience Replay (ER): armazena subconjunto de dados antigos
- Reservoir Sampling: garantia de uniformidade com memoria fixa
- Feature Replay: opcionalmente armazena hidden states (compacto)
"""

import torch
import random
from typing import Dict, Optional, Tuple, List
from collections import deque


class ReplayBuffer:
    """Buffer de replay com reservoir sampling.

    Armazena pares (input_ids, labels) de fases anteriores e permite
    misturar com o batch atual durante treinamento.

    Args:
        buffer_size: tamanho maximo do buffer (em amostras)
        mix_ratio: fracao do batch que vem do replay (0.0-1.0)
        device: device default (amostras armazenadas em CPU)
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        mix_ratio: float = 0.1,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.mix_ratio = mix_ratio
        self.device = device

        # Armazena em CPU para economizar VRAM
        self._input_ids: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []
        self._n_seen = 0

    @property
    def size(self) -> int:
        """Numero atual de amostras no buffer."""
        return len(self._input_ids)

    @property
    def is_empty(self) -> bool:
        """Buffer vazio."""
        return self.size == 0

    @torch.no_grad()
    def add(self, batch: Dict[str, torch.Tensor]) -> None:
        """Adiciona amostras ao buffer via reservoir sampling.

        Cada amostra no batch tem probabilidade buffer_size/n_seen
        de ser selecionada (garantia de uniformidade).

        Args:
            batch: dict com 'input_ids' [B, T] e 'labels' [B, T]
        """
        input_ids = batch["input_ids"].cpu()
        labels = batch["labels"].cpu()
        B = input_ids.shape[0]

        for i in range(B):
            self._n_seen += 1

            if self.size < self.buffer_size:
                # Buffer ainda nao cheio
                self._input_ids.append(input_ids[i].clone())
                self._labels.append(labels[i].clone())
            else:
                # Reservoir sampling: substitui com prob buffer_size/n_seen
                j = random.randint(0, self._n_seen - 1)
                if j < self.buffer_size:
                    self._input_ids[j] = input_ids[i].clone()
                    self._labels[j] = labels[i].clone()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: Optional[str] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Amostra um batch do buffer.

        Args:
            batch_size: tamanho do batch desejado
            device: device de destino (None = self.device)

        Returns:
            Dict com 'input_ids' e 'labels', ou None se buffer vazio
        """
        if self.is_empty:
            return None

        dev = device or self.device
        n = min(batch_size, self.size)
        indices = random.sample(range(self.size), n)

        input_ids = torch.stack([self._input_ids[i] for i in indices]).to(dev)
        labels = torch.stack([self._labels[i] for i in indices]).to(dev)

        return {"input_ids": input_ids, "labels": labels}

    def mix_batch(
        self,
        current_batch: Dict[str, torch.Tensor],
        device: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Mistura batch atual com amostras do replay buffer.

        Se mix_ratio=0.1 e batch_size=16, substitui ~1-2 amostras
        do batch atual por amostras do buffer.

        Args:
            current_batch: batch atual de treinamento
            device: device de destino

        Returns:
            Batch misturado
        """
        if self.is_empty or self.mix_ratio <= 0.0:
            return current_batch

        dev = device or self.device
        input_ids = current_batch["input_ids"].to(dev)
        labels = current_batch["labels"].to(dev)
        B = input_ids.shape[0]

        # Quantas amostras substituir
        n_replay = max(1, int(B * self.mix_ratio))
        n_replay = min(n_replay, self.size)

        # Amostras do buffer
        replay = self.sample(n_replay, dev)
        if replay is None:
            return current_batch

        replay_ids = replay["input_ids"]
        replay_labels = replay["labels"]

        # Trunca/pad se seq_len diferente
        T_cur = input_ids.shape[1]
        T_rep = replay_ids.shape[1]
        if T_rep > T_cur:
            replay_ids = replay_ids[:, :T_cur]
            replay_labels = replay_labels[:, :T_cur]
        elif T_rep < T_cur:
            pad = torch.zeros(
                n_replay, T_cur - T_rep,
                dtype=replay_ids.dtype,
                device=dev,
            )
            replay_ids = torch.cat([replay_ids, pad], dim=1)
            replay_labels = torch.cat([replay_labels, pad], dim=1)

        # Substitui ultimas n_replay amostras do batch
        mixed_ids = torch.cat([input_ids[:B - n_replay], replay_ids], dim=0)
        mixed_labels = torch.cat([labels[:B - n_replay], replay_labels], dim=0)

        return {"input_ids": mixed_ids, "labels": mixed_labels}

    def clear(self) -> None:
        """Limpa buffer."""
        self._input_ids.clear()
        self._labels.clear()
        self._n_seen = 0

    def state_dict(self) -> Dict:
        """Estado para checkpoint."""
        return {
            "input_ids": self._input_ids[:100],  # Limita para checkpoint
            "labels": self._labels[:100],
            "n_seen": self._n_seen,
            "buffer_size": self.buffer_size,
            "mix_ratio": self.mix_ratio,
        }

    def load_state_dict(self, state: Dict) -> None:
        """Carrega estado de checkpoint."""
        self._input_ids = state.get("input_ids", [])
        self._labels = state.get("labels", [])
        self._n_seen = state.get("n_seen", 0)

    def stats(self) -> Dict[str, float]:
        """Estatisticas do buffer."""
        return {
            "replay/buffer_size": self.size,
            "replay/max_size": self.buffer_size,
            "replay/fill_ratio": self.size / max(self.buffer_size, 1),
            "replay/total_seen": self._n_seen,
            "replay/mix_ratio": self.mix_ratio,
        }
