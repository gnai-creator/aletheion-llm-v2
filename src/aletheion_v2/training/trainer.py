"""
Trainer: Loop de treinamento para AletheionV2.

Funcionalidades:
- Treinamento com loss composta (CE + VARO + VI + MAD)
- Gradient clipping
- LR scheduling (warmup + cosine)
- Loss weight annealing
- Logging periodico
- Avaliacao periodica
- Checkpoint save/load
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import time
import os

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model
from aletheion_v2.loss.composite_loss import AletheionV2Loss
from aletheion_v2.training.scheduler import WarmupCosineScheduler


class Trainer:
    """Trainer para AletheionV2.

    Args:
        model: AletheionV2Model
        config: AletheionV2Config
        train_loader: DataLoader de treinamento
        eval_loader: DataLoader de avaliacao (opcional)
        device: Device (cpu/cuda)
    """

    def __init__(
        self,
        model: AletheionV2Model,
        config: AletheionV2Config,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device

        # Loss
        self.criterion = AletheionV2Loss(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Steps totais
        self.total_steps = config.max_epochs * len(train_loader)

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=self.total_steps,
            lr_max=config.learning_rate,
        )

        # Estado
        self.global_step = 0
        self.best_eval_loss = float("inf")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Treina uma epoch.

        Args:
            epoch: Numero da epoch

        Returns:
            Dict com metricas medias da epoch
        """
        self.model.train()
        total_losses = {}
        n_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward
            output = self.model(input_ids, return_tomography=True)

            # Loss
            G = None
            if output.tomography is not None:
                G = self.model.epistemic_head.get_metric_tensor()

            losses = self.criterion(
                output.logits, labels,
                tomography=output.tomography,
                G=G,
                step=self.global_step,
                total_steps=self.total_steps,
            )

            # Backward
            self.optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

            # Step
            self.optimizer.step()
            lr = self.scheduler.step()

            # Acumula metricas
            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                total_losses[k] = total_losses.get(k, 0.0) + val
            total_losses["lr"] = total_losses.get("lr", 0.0) + lr
            n_batches += 1

            # Logging
            if self.global_step % self.config.log_interval == 0:
                self._log_step(epoch, losses, lr)

            # Eval
            if (
                self.eval_loader is not None
                and self.global_step % self.config.eval_interval == 0
                and self.global_step > 0
            ):
                eval_loss = self.evaluate()
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss

            self.global_step += 1

        # Media
        avg = {k: v / n_batches for k, v in total_losses.items()}
        return avg

    @torch.no_grad()
    def evaluate(self) -> float:
        """Avalia no eval_loader.

        Returns:
            loss media de avaliacao
        """
        if self.eval_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            output = self.model(input_ids, return_tomography=True)

            G = None
            if output.tomography is not None:
                G = self.model.epistemic_head.get_metric_tensor()

            losses = self.criterion(
                output.logits, labels,
                tomography=output.tomography,
                G=G,
                step=self.global_step,
                total_steps=self.total_steps,
            )
            total_loss += losses["total"].item()
            n_batches += 1

        self.model.train()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  [EVAL] loss={avg_loss:.4f} (best={self.best_eval_loss:.4f})")
        return avg_loss

    def train(self) -> Dict[str, Any]:
        """Loop completo de treinamento.

        Returns:
            Dict com historico de metricas
        """
        history = {"epochs": []}
        print(f"[TRAIN] Iniciando treinamento: {self.config.max_epochs} epochs, "
              f"{self.total_steps} steps totais")
        print(f"[TRAIN] Params: {self.model.count_parameters()}")

        start_time = time.time()

        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()
            avg_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            avg_metrics["epoch"] = epoch
            avg_metrics["epoch_time"] = epoch_time
            history["epochs"].append(avg_metrics)

            print(
                f"[EPOCH {epoch}] loss={avg_metrics.get('total', 0):.4f} "
                f"ce={avg_metrics.get('ce', 0):.4f} "
                f"varo={avg_metrics.get('varo', 0):.4f} "
                f"vi={avg_metrics.get('vi', 0):.4f} "
                f"mad={avg_metrics.get('mad', 0):.4f} "
                f"time={epoch_time:.1f}s"
            )

        total_time = time.time() - start_time
        history["total_time"] = total_time
        print(f"[TRAIN] Completo em {total_time:.1f}s")

        return history

    def _log_step(
        self,
        epoch: int,
        losses: Dict[str, torch.Tensor],
        lr: float,
    ) -> None:
        """Log de um step."""
        total = losses["total"].item() if isinstance(losses["total"], torch.Tensor) else losses["total"]
        ce = losses.get("ce", torch.tensor(0.0))
        ce_val = ce.item() if isinstance(ce, torch.Tensor) else ce
        anneal = losses.get("annealing", torch.tensor(0.0))
        anneal_val = anneal.item() if isinstance(anneal, torch.Tensor) else anneal
        print(
            f"  step={self.global_step} epoch={epoch} "
            f"loss={total:.4f} ce={ce_val:.4f} "
            f"anneal={anneal_val:.2f} lr={lr:.2e}"
        )

    def save_checkpoint(self, path: str) -> None:
        """Salva checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config,
        }, path)
        print(f"[SAVE] Checkpoint salvo em {path}")

    def load_checkpoint(self, path: str) -> None:
        """Carrega checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        print(f"[LOAD] Checkpoint carregado de {path} (step={self.global_step})")
