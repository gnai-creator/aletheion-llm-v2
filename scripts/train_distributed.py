"""
Script de lancamento de treinamento distribuido.

Uso:
    # Single GPU
    python scripts/train_distributed.py --config configs/scaling/125m.yaml

    # Single GPU com dados especificos
    python scripts/train_distributed.py --config configs/scaling/350m_rtx4090.yaml --data-dir data/125m

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 scripts/train_distributed.py --config configs/scaling/7b.yaml

    # Multi-node (2 nodes x 8 GPUs)
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
        --master_addr=<IP> --master_port=29500 \
        scripts/train_distributed.py --config configs/scaling/70b.yaml

    # Resume de checkpoint
    python scripts/train_distributed.py --config configs/scaling/7b.yaml --resume checkpoints/step_5000.pt
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model
from aletheion_v2.training.trainer_distributed import DistributedTrainer
from aletheion_v2.training.data_pipeline import create_dataloader_from_config
from aletheion_v2.training.distributed import setup_distributed


def _detect_vocab_size(data_dir: str) -> int:
    """Detecta vocab_size do metadata dos dados.

    Le metadata.json no diretorio de dados para obter o vocab_size
    usado na tokenizacao. Evita mismatch entre modelo e dados.

    Args:
        data_dir: Diretorio de dados

    Returns:
        vocab_size ou 0 se nao encontrado
    """
    meta_path = Path(data_dir) / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("vocab_size", 0)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Treinamento AletheionV2")
    parser.add_argument("--config", required=True, help="Caminho do YAML de config")
    parser.add_argument("--resume", default="", help="Checkpoint para resume")
    parser.add_argument("--data-dir", default="", help="Override do diretorio de dados")
    parser.add_argument("--eval-data-dir", default="", help="Dados de avaliacao")
    parser.add_argument("--override", nargs="*", help="Overrides: key=value")
    args = parser.parse_args()

    # Carrega config
    config = AletheionV2Config.from_yaml(args.config)

    # Overrides de CLI
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.override:
        for ov in args.override:
            key, val = ov.split("=", 1)
            if hasattr(config, key):
                field_type = type(getattr(config, key))
                if field_type == bool:
                    setattr(config, key, val.lower() in ("true", "1", "yes"))
                else:
                    setattr(config, key, field_type(val))

    # Auto-deteccao de vocab_size dos dados
    data_vocab = _detect_vocab_size(config.data_dir)
    if data_vocab > 0 and data_vocab != config.vocab_size:
        print(f"[WARN] vocab_size do config ({config.vocab_size}) difere dos "
              f"dados ({data_vocab}). Ajustando para {data_vocab}.")
        config.vocab_size = data_vocab

    # Setup distribuido (para pegar rank antes de criar dataloader)
    dist_info = setup_distributed(config)
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    is_main = dist_info["is_main"]

    if is_main:
        print(f"[CONFIG] {args.config}")
        print(f"[CONFIG] d_model={config.d_model}, n_layers={config.n_layers}, "
              f"n_heads={config.n_heads}")
        est_params = config.total_params_estimate
        print(f"[CONFIG] ~{est_params:,} params estimados")
        print(f"[CONFIG] vocab_size={config.vocab_size}, seq_len={config.max_seq_len}")
        print(f"[CONFIG] total_tokens={config.total_tokens:,}")
        print(f"[CONFIG] batch={config.batch_size} x accum={config.gradient_accumulation_steps} "
              f"x world={world_size}")
        if config.enable_ewc:
            print(f"[CONFIG] EWC: lambda={config.ewc_lambda}, online={config.ewc_online}")
        if config.enable_replay:
            print(f"[CONFIG] Replay: buffer={config.replay_buffer_size}, "
                  f"mix={config.replay_mix_ratio}")

    # Cria modelo
    model = AletheionV2Model(config)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] Parametros reais: {total_params:,}")

        # Estimativa de memoria para GPU
        param_bytes = total_params * 2  # bf16
        optim_bytes = total_params * 12  # AdamW fp32 states
        total_gb = (param_bytes + optim_bytes) / (1024 ** 3)
        print(f"[MODEL] Memoria estimada (modelo+optimizer): {total_gb:.1f} GB")

    # Cria dataloaders
    train_loader = create_dataloader_from_config(
        config, split="train", rank=rank, world_size=world_size,
    )

    eval_loader = None
    if args.eval_data_dir:
        eval_config = AletheionV2Config.from_yaml(args.config)
        eval_config.data_dir = args.eval_data_dir
        eval_loader = create_dataloader_from_config(
            eval_config, split="validation", rank=rank, world_size=world_size,
        )

    # Cria trainer
    trainer = DistributedTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Treina
    history = trainer.train()

    # Consolida fase para continual learning
    if config.enable_ewc:
        trainer.consolidate_phase()

    if is_main:
        print(f"\n[DONE] Treinamento completo.")
        print(f"  Total time: {history['total_time']:.0f}s")
        print(f"  Checkpoints em: {config.save_dir}/")


if __name__ == "__main__":
    main()
