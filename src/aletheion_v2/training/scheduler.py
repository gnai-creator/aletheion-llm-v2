"""
Learning Rate Scheduler com warmup + cosine decay.

Tambem faz annealing dos pesos de loss epistemica
(warmup -> ramp -> full).
"""

import torch
import math
from typing import Optional


class WarmupCosineScheduler:
    """Scheduler com warmup linear + cosine decay.

    LR sobe linearmente de 0 ate lr_max durante warmup_steps,
    depois decai via cosine ate lr_min.

    Args:
        optimizer: Otimizador PyTorch
        warmup_steps: Steps de warmup linear
        total_steps: Total de steps de treinamento
        lr_max: LR maximo (apos warmup)
        lr_min: LR minimo (no final)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_max: float = 3e-4,
        lr_min: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_step = 0

    def get_lr(self) -> float:
        """Computa LR para step atual."""
        if self.current_step < self.warmup_steps:
            # Warmup linear
            return self.lr_max * (self.current_step / max(self.warmup_steps, 1))

        # Cosine decay
        progress = (self.current_step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine

    def step(self) -> float:
        """Avanca um step e atualiza LR do optimizer.

        Returns:
            lr: LR atual
        """
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1
        return lr


class LossWeightAnnealer:
    """Annealing dos pesos das losses epistemicas.

    Fases:
    1. warmup (0 -> warmup_end): peso = 0 (so CE)
    2. ramp (warmup_end -> ramp_end): peso ramp linear 0 -> 1
    3. full (ramp_end -> end): peso = 1

    Args:
        warmup_fraction: Fracao do treino com peso = 0
        ramp_fraction: Fracao do treino onde ramp termina
        total_steps: Total de steps
    """

    def __init__(
        self,
        warmup_fraction: float = 0.1,
        ramp_fraction: float = 0.5,
        total_steps: int = 10000,
    ):
        self.warmup_end = int(total_steps * warmup_fraction)
        self.ramp_end = int(total_steps * ramp_fraction)

    def get_weight(self, step: int) -> float:
        """Retorna peso de annealing para step."""
        if step < self.warmup_end:
            return 0.0
        if step >= self.ramp_end:
            return 1.0
        progress = (step - self.warmup_end) / max(
            self.ramp_end - self.warmup_end, 1
        )
        return progress
