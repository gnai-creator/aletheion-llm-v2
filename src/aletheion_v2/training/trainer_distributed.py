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
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

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

        # Force epistemic head to fp32 (bf16 causes NaN in tomography)
        if hasattr(model, "epistemic_head"):
            model.epistemic_head.float()
            if self.is_main:
                print("[FP32] epistemic_head forçado para fp32")

        # Force all LayerNorm to fp32 (bf16 accumulation over 24 layers
        # causes hidden_states overflow — LayerNorm in fp32 prevents this)
        ln_count = 0
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                module.float()
                ln_count += 1
        if self.is_main:
            print(f"[FP32] {ln_count} LayerNorm forçados para fp32")

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
        self.evals_without_improvement = 0
        self.early_stopping_patience = config.early_stopping_patience

        # Logging
        self.wandb_run = None
        self.tb_writer = None
        self.file_logger = None
        if self.is_main:
            self._init_logging()

        # Diretorio de checkpoints
        self.save_dir = Path(config.save_dir)
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self._init_file_logger()

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

    def _init_file_logger(self) -> None:
        """Inicializa logger em arquivo texto com todas as metricas."""
        log_path = self.save_dir / "train.log"
        self.file_logger = logging.getLogger(f"aletheion_train_{id(self)}")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.handlers.clear()
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self.file_logger.addHandler(fh)
        self.file_logger.info(f"=== Training started: {datetime.now().isoformat()} ===")
        self.file_logger.info(f"Config: {self.config}")

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Loga metricas em wandb, tensorboard e arquivo."""
        if not self.is_main:
            return

        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)

        if self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)

        # File logger: todas as metricas em formato parseable
        if self.file_logger:
            parts = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in sorted(metrics.items())]
            self.file_logger.info(" | ".join(parts))

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
            output = self.model(input_ids, return_tomography=self.config.enable_tomography)

            # Skip step if logits or hidden_states are NaN/Inf/dangerously large
            hs_bad = (torch.isnan(output.hidden_states).any() or
                      torch.isinf(output.hidden_states).any())
            logits_bad = (torch.isnan(output.logits).any() or
                          torch.isinf(output.logits).any())
            # Also skip if hidden_states magnitude is dangerously high
            hs_max = output.hidden_states.abs().max()
            hs_danger = hs_max > 1000  # Normal range is <100 for d_model=1024
            if logits_bad or hs_bad or hs_danger:
                if self.is_main:
                    print(f"  [SKIP] step~{self.global_step} "
                          f"logits_bad={logits_bad} hs_bad={hs_bad} "
                          f"hs_max={hs_max.item():.0f}")
                self.optimizer.zero_grad()
                return {"total": float('nan'), "ce": float('nan'), "nan_skipped": 1.0}

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
                hidden_states=output.hidden_states,
            )

            loss = losses["total"] / self.config.gradient_accumulation_steps

            # EWC: adiciona penalidade
            if self.ewc is not None and self.ewc.has_fisher:
                ewc_loss = self.ewc(self.raw_model)
                ewc_loss = ewc_loss / self.config.gradient_accumulation_steps
                loss = loss + ewc_loss
                losses["ewc"] = ewc_loss * self.config.gradient_accumulation_steps

        # NaN guard: skip backward if loss is NaN to prevent weight corruption
        if torch.isnan(loss) or torch.isinf(loss):
            self.optimizer.zero_grad()
            metrics = {k: float('nan') for k in losses}
            tomo = output.tomography
            if tomo is not None:
                if tomo.confidence is not None:
                    metrics["avg_conf"] = float('nan')
                if hasattr(tomo, 'phi') and tomo.phi is not None:
                    metrics["avg_phi"] = float('nan')
            return metrics

        # Backward
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in losses.items()
        }

        # Extrai TODAS as metricas epistemicas da tomografia
        tomo = output.tomography
        if tomo is not None:
            # --- Core ---
            if tomo.q1 is not None:
                metrics["avg_q1"] = tomo.q1.mean().item()
            if tomo.q2 is not None:
                metrics["avg_q2"] = tomo.q2.mean().item()
            if tomo.confidence is not None:
                metrics["avg_confidence"] = tomo.confidence.mean().item()
            if tomo.metric_distance is not None:
                metrics["avg_geodesic_distance"] = tomo.metric_distance.mean().item()
            if tomo.directional_dim is not None:
                metrics["avg_dim_d"] = tomo.directional_dim.mean().item()
            if tomo.phi_total is not None:
                metrics["avg_phi"] = tomo.phi_total.mean().item()
            if tomo.phi_components is not None:
                phi_c = tomo.phi_components.mean(dim=(0, 1))  # [4]
                metrics["phi_dim"] = phi_c[0].item()
                metrics["phi_disp"] = phi_c[1].item()
                metrics["phi_ent"] = phi_c[2].item()
                metrics["phi_conf"] = phi_c[3].item()
            if tomo.vi_severity is not None:
                metrics["avg_vi_severity"] = tomo.vi_severity.mean().item()
            if tomo.temperature is not None:
                metrics["avg_temperature"] = tomo.temperature.mean().item()
            if tomo.drm_coords is not None:
                coords = tomo.drm_coords.mean(dim=(0, 1))  # [5]
                for i in range(min(coords.shape[0], 5)):
                    metrics[f"drm_coord_{i}"] = coords[i].item()
                metrics["drm_coord_std"] = tomo.drm_coords.std().item()

            # --- Tier 1: Eidos ---
            if tomo.eidos_weights is not None:
                metrics["avg_eidos_weight"] = tomo.eidos_weights.mean().item()
            if tomo.axis_balance is not None:
                bal = tomo.axis_balance.mean(dim=(0, 1))  # [5]
                for i in range(min(bal.shape[0], 5)):
                    metrics[f"axis_balance_{i}"] = bal[i].item()

            # --- Tier 1: Filosofia3 ---
            if tomo.conflict_intensity is not None:
                metrics["avg_conflict"] = tomo.conflict_intensity.mean().item()
            if tomo.mode_probs is not None:
                mp = tomo.mode_probs.mean(dim=(0, 1))  # [4]
                for i in range(min(mp.shape[0], 4)):
                    metrics[f"mode_prob_{i}"] = mp[i].item()

            # --- Tier 1: Consciousness ---
            if tomo.mood is not None:
                metrics["avg_mood"] = tomo.mood.mean().item()
            if tomo.curiosity is not None:
                metrics["avg_curiosity"] = tomo.curiosity.mean().item()
            if tomo.energy is not None:
                metrics["avg_energy"] = tomo.energy.mean().item()
            if tomo.drives is not None:
                dr = tomo.drives.mean(dim=(0, 1))  # [3]
                metrics["drive_curiosity"] = dr[0].item()
                metrics["drive_mastery"] = dr[1].item()
                metrics["drive_autonomy"] = dr[2].item()

            # --- Tier 2: Grounding ---
            if tomo.task_confidence is not None:
                metrics["avg_task_confidence"] = tomo.task_confidence.mean().item()
            if tomo.ambiguity_level is not None:
                metrics["avg_ambiguity"] = tomo.ambiguity_level.mean().item()

            # --- Tier 2: Plasticity ---
            if tomo.plasticity_remaining is not None:
                metrics["avg_plasticity"] = tomo.plasticity_remaining.mean().item()
            if tomo.gate_value is not None:
                metrics["avg_gate"] = tomo.gate_value.mean().item()

            # --- Tier 2: MPL ---
            if tomo.frontier_score is not None:
                metrics["avg_frontier"] = tomo.frontier_score.mean().item()

            # --- Tier 3: MOPsi ---
            if tomo.psi is not None:
                metrics["avg_psi"] = tomo.psi.mean().item()
            if tomo.mediated_score is not None:
                metrics["avg_mediated"] = tomo.mediated_score.mean().item()

            # --- Tier 3: CausalState ---
            if tomo.state_gate is not None:
                metrics["avg_state_gate"] = tomo.state_gate.mean().item()

            # --- Tier 3: Metacognitive ---
            if tomo.divergence is not None:
                metrics["avg_divergence"] = tomo.divergence.mean().item()

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
            gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            if not math.isfinite(gn):
                # NaN/Inf grad norm: skip optimizer step to prevent weight corruption.
                # clip_grad_norm_ with NaN norm makes ALL grads NaN — must discard.
                self.optimizer.zero_grad(set_to_none=True)
                lr = self.scheduler.step()
                if self.is_main:
                    print(f"  [SKIP-GRAD] grad_norm={gn}, skipping optimizer step")
                return lr, gn
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            if not math.isfinite(gn):
                # NaN/Inf grad norm: skip optimizer step to prevent weight corruption.
                # clip_grad_norm_ with NaN norm makes ALL grads NaN — must discard.
                self.optimizer.zero_grad(set_to_none=True)
                lr = self.scheduler.step()
                if self.is_main:
                    print(f"  [SKIP-GRAD] grad_norm={gn}, skipping optimizer step")
                return lr, gn
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        lr = self.scheduler.step()
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
        tokens_at_start = self.tokens_seen  # Para tok/s correto no resume
        accum_metrics = {}
        accum_count = 0

        self.optimizer.zero_grad(set_to_none=True)

        if self.is_main and self.global_step > 0:
            print(f"[TRAIN] Resumindo do step {self.global_step}/{self.total_steps} "
                  f"({self.tokens_seen:,} tokens ja processados)")

        epoch = 0
        data_iter = iter(self.train_loader)

        nan_step = False
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

                # Detect NaN step
                if "nan_fallback" in metrics or (
                    isinstance(metrics.get("total"), float) and
                    metrics["total"] != metrics["total"]  # NaN check
                ):
                    nan_step = True

                # Acumula metricas
                for k, v in metrics.items():
                    accum_metrics[k] = accum_metrics.get(k, 0) + v
                accum_count += 1

            # Optimizer step (skip if NaN to prevent weight corruption)
            if nan_step:
                self.optimizer.zero_grad()
                lr = self.optimizer.param_groups[0]["lr"]
                grad_norm = 0.0
                nan_step = False
            else:
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
                elapsed = time.time() - start_time
                tokens_this_run = self.tokens_seen - tokens_at_start
                avg["tokens_per_sec"] = (
                    tokens_this_run / elapsed if elapsed > 0 else 0
                )

                if self.is_main:
                    phi_str = f" phi={avg.get('avg_phi', 0):.3f}" if "avg_phi" in avg else ""
                    conf_str = f" conf={avg.get('avg_confidence', 0):.3f}" if "avg_confidence" in avg else ""
                    stp_str = f" stp={avg.get('stp', 0):.4f}" if avg.get("stp", 0) > 0 else ""
                    print(
                        f"  step={self.global_step}/{self.total_steps} "
                        f"loss={avg.get('total', 0):.4f} "
                        f"ce={avg.get('ce', 0):.4f}{stp_str} "
                        f"lr={lr:.2e} "
                        f"gnorm={grad_norm:.2f} "
                        f"tok/s={avg['tokens_per_sec']:.0f}"
                        f"{phi_str}{conf_str} "
                        f"tokens={self.tokens_seen:,}"
                    )

                self._log_metrics(avg, self.global_step)
                history.append(avg)
                # Incremental save: training_log.json atualizado a cada log_interval
                if self.is_main and self.global_step % (self.config.log_interval * 10) == 0:
                    self._save_training_log(history, time.time() - start_time)
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
                    self.evals_without_improvement = 0
                    if self.is_main:
                        self.save_checkpoint("best.pt")
                else:
                    self.evals_without_improvement += 1

                # Early stopping
                if (
                    self.early_stopping_patience > 0
                    and self.evals_without_improvement >= self.early_stopping_patience
                ):
                    if self.is_main:
                        print(
                            f"[EARLY STOP] Nenhuma melhora por "
                            f"{self.evals_without_improvement} avaliacoes. "
                            f"Parando no step {self.global_step}."
                        )
                    break

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
            self._generate_plots()

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
                output = self.model(input_ids, return_tomography=self.config.enable_tomography)
                losses = self.criterion(output.logits, labels)

            total_loss += losses["total"].item()
            n += 1
            if n >= 50:  # Max 50 batches para eval rapido
                break

        avg = total_loss / max(n, 1)
        if self.is_main:
            print(f"  [EVAL] loss={avg:.4f} (best={self.best_eval_loss:.4f})")
            self._log_metrics({"eval/loss": avg}, self.global_step)
            if self.file_logger:
                self.file_logger.info(
                    f"[EVAL] step={self.global_step} eval_loss={avg:.6f} "
                    f"best={self.best_eval_loss:.6f}"
                )

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
        if "stp" in log:
            log["stp_loss"] = log.pop("stp")
        if "lr" in log:
            log["learning_rates"] = log.pop("lr")

        path = self.save_dir / "training_log.json"
        with open(path, "w") as f:
            json.dump(log, f, indent=2, default=str)
        print(f"  [LOG] {path}")

    def _generate_plots(self) -> None:
        """Gera graficos de treinamento automaticamente no final."""
        log_path = self.save_dir / "training_log.json"
        if not log_path.exists():
            print("[WARN] training_log.json nao encontrado, pulando plots")
            return

        try:
            import subprocess
            import sys
            plot_script = Path(__file__).parent.parent.parent.parent / "scripts" / "plot_training.py"
            if not plot_script.exists():
                print(f"[WARN] plot_training.py nao encontrado em {plot_script}")
                return

            plot_dir = self.save_dir / "plots"
            result = subprocess.run(
                [
                    sys.executable, str(plot_script),
                    "--log", str(log_path),
                    "--output", str(plot_dir),
                ],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                print(f"  [PLOT] Graficos gerados em {plot_dir}/")
            else:
                print(f"  [WARN] Plots falharam: {result.stderr[:200]}")
        except Exception as e:
            print(f"  [WARN] Nao foi possivel gerar plots: {e}")

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
