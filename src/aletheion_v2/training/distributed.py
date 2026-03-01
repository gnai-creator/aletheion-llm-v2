"""
Distributed Training: DDP, FSDP, mixed precision, gradient checkpointing.

Suporta:
- DDP (DistributedDataParallel): replica modelo em cada GPU
- FSDP (FullyShardedDataParallel): shard modelo entre GPUs (para 7B+)
- Mixed precision: bf16 (Ampere+) ou fp16
- Gradient checkpointing: troca memoria por compute
- Gradient accumulation: simula batches maiores
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional
from contextlib import contextmanager

from aletheion_v2.config import AletheionV2Config


def setup_distributed(config: AletheionV2Config) -> dict:
    """Inicializa ambiente distribuido.

    Detecta automaticamente se esta rodando via torchrun/launch.

    Returns:
        Dict com rank, local_rank, world_size, device
    """
    if not config.distributed:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": device,
            "is_main": True,
        }

    # Variaveis setadas pelo torchrun
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Inicializa process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=config.dist_backend,
            rank=rank,
            world_size=world_size,
        )

    # Seta device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "is_main": rank == 0,
    }


def cleanup_distributed() -> None:
    """Finaliza processo distribuido."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: torch.nn.Module,
    config: AletheionV2Config,
    device: str,
) -> torch.nn.Module:
    """Envolve modelo com DDP ou FSDP.

    Args:
        model: Modelo base
        config: Configuracao
        device: Device string

    Returns:
        Modelo wrapped para treinamento distribuido
    """
    model = model.to(device)

    # Gradient checkpointing
    if config.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    # torch.compile (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    if not config.distributed:
        return model

    if config.fsdp:
        return _wrap_fsdp(model, config, device)
    else:
        return _wrap_ddp(model, device)


def _wrap_ddp(
    model: torch.nn.Module,
    device: str,
) -> torch.nn.Module:
    """Envolve com DistributedDataParallel."""
    local_rank = int(device.split(":")[-1]) if "cuda" in device else 0
    return DDP(
        model,
        device_ids=[local_rank] if "cuda" in device else None,
        output_device=local_rank if "cuda" in device else None,
        find_unused_parameters=True,  # Necessario para DirectionalField
    )


def _wrap_fsdp(
    model: torch.nn.Module,
    config: AletheionV2Config,
    device: str,
) -> torch.nn.Module:
    """Envolve com FullyShardedDataParallel.

    FSDP shard pesos, gradientes e estados do otimizador entre GPUs.
    Essencial para modelos que nao cabem numa unica GPU (7B+).
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from aletheion_v2.core.transformer_block import TransformerBlock

    # Estrategia de sharding
    sharding_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    sharding = sharding_map.get(config.fsdp_sharding, ShardingStrategy.FULL_SHARD)

    # Mixed precision para FSDP
    mp_policy = None
    if config.mixed_precision == "bf16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif config.mixed_precision == "fp16":
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # Auto wrap policy: shard por TransformerBlock
    auto_wrap = transformer_auto_wrap_policy(
        transformer_layer_cls={TransformerBlock}
    )

    return FSDP(
        model,
        sharding_strategy=sharding,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap,
        device_id=torch.cuda.current_device() if "cuda" in device else None,
    )


def _enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Habilita gradient checkpointing nos TransformerBlocks.

    Troca compute por memoria: re-computa ativacoes no backward
    em vez de guardar todas. Reduz ~60% de uso de memoria.
    """
    from aletheion_v2.core.transformer_block import TransformerBlock

    for module in model.modules():
        if isinstance(module, TransformerBlock):
            module._orig_forward = module.forward

            def checkpointed_forward(self, *args, **kwargs):
                return torch.utils.checkpoint.checkpoint(
                    self._orig_forward, *args, use_reentrant=False, **kwargs
                )

            import types
            module.forward = types.MethodType(checkpointed_forward, module)


def get_mixed_precision_context(config: AletheionV2Config):
    """Retorna context manager para mixed precision.

    Returns:
        torch.amp.autocast context ou nullcontext
    """
    if config.mixed_precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    elif config.mixed_precision == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        from contextlib import nullcontext
        return nullcontext()


def get_grad_scaler(config: AletheionV2Config) -> Optional[torch.amp.GradScaler]:
    """Retorna GradScaler para fp16 (nao necessario para bf16).

    Returns:
        GradScaler ou None
    """
    if config.mixed_precision == "fp16":
        return torch.amp.GradScaler("cuda")
    return None
