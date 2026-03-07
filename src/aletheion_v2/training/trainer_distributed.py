"""
Trainer Distribuido: Loop de treinamento para escala 1M-162B.

Suporta:
- DDP/FSDP para multi-GPU/multi-node
- Mixed precision (bf16/fp16)
- Gradient accumulation
- Gradient checkpointing
- Wandb e TensorBoard logging
- Checkpoint save/resume robusto
- Token-based training (total_tokens em vez de epochs)
"""

import os
import time
import json
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
from pathlib import Path

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model
from aletheion_v2.loss.composite_loss import AletheionV2Loss
from aletheion_v2.training.scheduler import WarmupCosineScheduler
from aletheion_v2.training.ewc import EWCRegularizer
from aletheion_v2.training.replay_buffer import ReplayBuffer
from aletheion_v2.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    get_mixed_precision_context,
    get_grad_scaler,
)


class DistributedTrainer:
    """Trainer escalavel para 1M-162B parametros.

    Args:
        config: AletheionV2Config
        model: AletheionV2Model (unwrapped)
        train_loader: DataLoader de treinamento
        eval_loader: DataLoader de avaliacao
    """

    def __init__(
        self,
        config: AletheionV2Config,
        model: AletheionV2Model,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ):
        self.config = config

        # Setup distribuido
        self.dist_info = setup_distributed(config)
        self.device = self.dist_info["device"]
        self.is_main = self.dist_info["is_main"]
        self.rank = self.dist_info["rank"]
        self.world_size = self.dist_info["world_size"]

        # Wrap modelo
        self.model = wrap_model_ddp(model, config, self.device)
        self.raw_model = model  # Referencia sem wrapper

        # Loss
        self.criterion = AletheionV2Loss(config)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Tokens por step
        self.tokens_per_step = (
            config.batch_size
            * config.max_seq_len
            * config.gradient_accumulation_steps
            * self.world_size
        )

        # Total de steps
        if config.total_tokens > 0:
            self.total_steps = config.total_tokens // self.tokens_per_step
        else:
            self.total_steps = config.max_epochs * len(train_loader)

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=self.total_steps,
            lr_max=config.learning_rate,
        )

        # Mixed precision
        self.amp_context = get_mixed_precision_context(config)
        self.grad_scaler = get_grad_scaler(config)

        # Data
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Continual Learning
        self.ewc = None
        if config.enable_ewc:
            self.ewc = EWCRegularizer(
                lambda_ewc=config.ewc_lambda,
                online=config.ewc_online,
                gamma=config.ewc_gamma,
            )

        self.replay_buffer = None
        if config.enable_replay:
            self.replay_buffer = ReplayBuffer(
                buffer_size=config.replay_buffer_size,
                mix_ratio=config.replay_mix_ratio,
            )

        # Estado
        self.global_step = 0
        self.tokens_seen = 0
        self.best_eval_loss = float("inf")

        # Logging
        self.wandb_run = None
        self.tb_writer = None
        if self.is_main:
            self._init_logging()

        # Diretorio de checkpoints
        self.save_dir = Path(config.save_dir)
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Cria optimizer com weight decay seletivo.

        Bias e LayerNorm nao recebem weight decay.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "ln" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _init_logging(self) -> None:
        """Inicializa wandb e/ou tensorboard."""
        if self.config.wandb_project:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name or None,
                    config=vars(self.config) if hasattr(self.config, '__dict__')
                    else {},
                )
            except ImportError:
                print("[WARN] wandb nao instalado, logging desabilitado")

        if self.config.tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.config.tensorboard_dir)
            except ImportError:
                print("[WARN] tensorboard nao instalado")

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Loga metricas."""
        if not self.is_main:
            return

        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)

        if self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)

    def _should_save(self) -> bool:
        """Verifica se deve salvar checkpoint."""
        if not self.is_main:
            return False
        if self.config.save_interval <= 0:
            return False
        return self.global_step % self.config.save_interval == 0

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Executa um step de treinamento (com gradient accumulation).

        Returns:
            Dict com metricas do step
        """
        self.model.train()

        # Replay: armazena e mistura batch
        if self.replay_buffer is not None:
            self.replay_buffer.add(batch)
            batch = self.replay_buffer.mix_batch(batch, str(self.device))

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward com mixed precision
        with self.amp_context:
            output = self.model(input_ids, return_tomography=True)

            # Metric tensor G
            G = None
            raw = self.raw_model
            if hasattr(raw, "epistemic_head"):
                G = raw.epistemic_head.get_metric_tensor()

            losses = self.criterion(
                output.logits, labels,
                tomography=output.tomography,
                G=G,
                step=self.global_step,
                total_steps=self.total_steps,
            )

            loss = losses["total"] / self.config.gradient_accumulation_steps

            # EWC: adiciona penalidade
            if self.ewc is not None and self.ewc.has_fisher:
                ewc_loss = self.ewc(self.raw_model)
                ewc_loss = ewc_loss / self.config.gradient_accumulation_steps
                loss = loss + ewc_loss
                losses["ewc"] = ewc_loss * self.config.gradient_accumulation_steps

        # Backward
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in losses.items()
        }

        # Extrai metricas epistemicas da tomografia
        tomo = output.tomography
        if tomo is not None:
            if tomo.q1 is not None:
                metrics["avg_q1"] = tomo.q1.mean().item()
            if tomo.q2 is not None:
                metrics["avg_q2"] = tomo.q2.mean().item()
            if tomo.confidence is not None:
                metrics["avg_confidence"] = tomo.confidence.mean().item()
            if tomo.phi_total is not None:
                metrics["avg_phi"] = tomo.phi_total.mean().item()
            if tomo.metric_distance is not None:
                metrics["avg_geodesic_distance"] = tomo.metric_distance.mean().item()
            if tomo.directional_dim is not None:
                metrics["avg_dim_d"] = tomo.directional_dim.mean().item()
            if tomo.vi_severity is not None:
                metrics["avg_vi_severity"] = tomo.vi_severity.mean().item()

        return metrics

    def _optimizer_step(self) -> tuple:
        """Executa optimizer step (apos accumulation).

        Returns:
            (lr, grad_norm)
        """
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        lr = self.scheduler.step()
        gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        return lr, gn

    def train(self) -> Dict[str, Any]:
        """Loop de treinamento principal.

        Returns:
            Historico de metricas
        """
        if self.is_main:
            param_count = sum(p.numel() for p in self.raw_model.parameters())
            print(f"[TRAIN] Modelo: {param_count:,} parametros")
            print(f"[TRAIN] Tokens por step: {self.tokens_per_step:,}")
            print(f"[TRAIN] Total steps: {self.total_steps:,}")
            if self.config.total_tokens > 0:
                print(f"[TRAIN] Total tokens: {self.config.total_tokens:,}")
            print(f"[TRAIN] Device: {self.device}, World: {self.world_size}")
            print(f"[TRAIN] Mixed precision: {self.config.mixed_precision}")

        history = []
        start_time = time.time()
        accum_metrics = {}
        accum_count = 0

        self.optimizer.zero_grad(set_to_none=True)

        epoch = 0
        data_iter = iter(self.train_loader)

        while self.global_step < self.total_steps:
            # Gradient accumulation loop
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                metrics = self._train_step(batch)

                # Acumula metricas
                for k, v in metrics.items():
                    accum_metrics[k] = accum_metrics.get(k, 0) + v
                accum_count += 1

            # Optimizer step
            lr, grad_norm = self._optimizer_step()

            # Atualiza contadores
            self.global_step += 1
            self.tokens_seen += self.tokens_per_step

            # Log
            if self.global_step % self.config.log_interval == 0:
                avg = {k: v / accum_count for k, v in accum_metrics.items()}
                avg["lr"] = lr
                avg["grad_norm"] = grad_norm
                avg["step"] = self.global_step
                avg["tokens_seen"] = self.tokens_seen
                avg["epoch"] = epoch
                avg["tokens_per_sec"] = (
                    self.tokens_seen / (time.time() - start_time)
                )

                if self.is_main:
                    phi_str = f" phi={avg.get('avg_phi', 0):.3f}" if "avg_phi" in avg else ""
                    conf_str = f" conf={avg.get('avg_confidence', 0):.3f}" if "avg_confidence" in avg else ""
                    print(
                        f"  step={self.global_step}/{self.total_steps} "
                        f"loss={avg.get('total', 0):.4f} "
                        f"ce={avg.get('ce', 0):.4f} "
                        f"lr={lr:.2e} "
                        f"gnorm={grad_norm:.2f} "
                        f"tok/s={avg['tokens_per_sec']:.0f}"
                        f"{phi_str}{conf_str} "
                        f"tokens={self.tokens_seen:,}"
                    )

                self._log_metrics(avg, self.global_step)
                history.append(avg)
                accum_metrics = {}
                accum_count = 0

            # Eval
            if (
                self.eval_loader is not None
                and self.global_step % self.config.eval_interval == 0
            ):
                eval_loss = self.evaluate()
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    if self.is_main:
                        self.save_checkpoint("best.pt")

            # Save
            if self._should_save():
                self.save_checkpoint(f"step_{self.global_step}.pt")
                self._cleanup_old_checkpoints()

        total_time = time.time() - start_time
        if self.is_main:
            print(
                f"[TRAIN] Completo: {self.global_step} steps, "
                f"{self.tokens_seen:,} tokens, {total_time:.0f}s"
            )
            self.save_checkpoint("final.pt")
            self._save_training_log(history, total_time)

        cleanup_distributed()
        return {"history": history, "total_time": total_time}

    @torch.no_grad()
    def evaluate(self) -> float:
        """Avalia no eval_loader."""
        self.model.eval()
        total_loss = 0
        n = 0

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with self.amp_context:
                output = self.model(input_ids, return_tomography=False)
                losses = self.criterion(output.logits, labels)

            total_loss += losses["total"].item()
            n += 1
            if n >= 50:  # Max 50 batches para eval rapido
                break

        avg = total_loss / max(n, 1)
        if self.is_main:
            print(f"  [EVAL] loss={avg:.4f} (best={self.best_eval_loss:.4f})")
            self._log_metrics({"eval/loss": avg}, self.global_step)

        self.model.train()
        return avg

    def consolidate_phase(self) -> None:
        """Consolida fase de treinamento para continual learning.

        Computa Fisher Information Matrix e preserva parametros.
        """
        if self.ewc is not None:
            if self.is_main:
                print("[CL] Computando Fisher Information Matrix...")
            self.ewc.compute_fisher(
                self.raw_model,
                self.train_loader,
                num_samples=self.config.ewc_fisher_samples,
                device=str(self.device),
                amp_context=self.amp_context,
            )
            if self.is_main:
                stats = self.ewc.get_importance_stats()
                avg = stats.get("fisher/avg_importance", 0)
                print(f"[CL] Fisher computada (avg={avg:.6f}, "
                      f"phases={stats.get('fisher/num_phases', 0)})")

    def save_checkpoint(self, name: str) -> None:
        """Salva checkpoint."""
        if not self.is_main:
            return

        path = self.save_dir / name
        state = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler_step": self.scheduler.current_step,
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config,
        }
        if self.grad_scaler is not None:
            state["grad_scaler"] = self.grad_scaler.state_dict()
        if self.ewc is not None:
            state["ewc"] = self.ewc.state_dict_ewc()
        if self.replay_buffer is not None:
            state["replay_buffer"] = self.replay_buffer.state_dict()

        torch.save(state, str(path))
        print(f"  [SAVE] {path}")

    def load_checkpoint(self, path: str) -> None:
        """Carrega checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.current_step = state["scheduler_step"]
        self.global_step = state["global_step"]
        self.tokens_seen = state["tokens_seen"]
        self.best_eval_loss = state.get("best_eval_loss", float("inf"))
        if self.grad_scaler and "grad_scaler" in state:
            self.grad_scaler.load_state_dict(state["grad_scaler"])
        if self.ewc is not None and "ewc" in state:
            self.ewc.load_state_dict_ewc(state["ewc"])
        if self.replay_buffer is not None and "replay_buffer" in state:
            self.replay_buffer.load_state_dict(state["replay_buffer"])
        if self.is_main:
            print(f"  [LOAD] {path} (step={self.global_step})")

    def _save_training_log(self, history: List[Dict], total_time: float) -> None:
        """Salva log de treinamento em JSON para visualizacao."""
        if not self.is_main or not history:
            return

        # Reorganiza para formato por metrica (listas)
        log = {
            "total_time": total_time,
            "total_steps": self.global_step,
            "total_tokens": self.tokens_seen,
        }

        # Transpoe de lista de dicts para dict de listas
        all_keys = set()
        for entry in history:
            all_keys.update(entry.keys())

        for key in sorted(all_keys):
            vals = [entry.get(key, None) for entry in history]
            # Remove Nones do final
            while vals and vals[-1] is None:
                vals.pop()
            if vals:
                log[key] = vals

        # Renomeia para compatibilidade com plot_training.py
        if "total" in log:
            log["train_losses"] = log.pop("total")
        if "ce" in log:
            log["ce_loss"] = log.pop("ce")
        if "varo" in log:
            log["varo_loss"] = log.pop("varo")
        if "vi" in log:
            log["vi_loss"] = log.pop("vi")
        if "mad" in log:
            log["mad_loss"] = log.pop("mad")
        if "lr" in log:
            log["learning_rates"] = log.pop("lr")

        path = self.save_dir / "training_log.json"
        with open(path, "w") as f:
            json.dump(log, f, indent=2, default=str)
        print(f"  [LOG] {path}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove checkpoints antigos alem do limite."""
        if self.config.save_total_limit <= 0:
            return
        ckpts = sorted(
            self.save_dir.glob("step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(ckpts) > self.config.save_total_limit:
            old = ckpts.pop(0)
            old.unlink()
