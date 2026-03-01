"""
Script auxiliar: treina e gera graficos automaticamente.

Uso:
    python scripts/train_and_plot.py --config configs/scaling/1m.yaml --data-dir data/1m
    python scripts/train_and_plot.py --config configs/scaling/10m.yaml --data-dir data/10m

Equivale a rodar train_distributed.py seguido de plot_training.py.
"""

import sys
import subprocess
from pathlib import Path


def main():
    # Repassa todos os args para train_distributed.py
    scripts_dir = Path(__file__).parent
    train_script = scripts_dir / "train_distributed.py"
    plot_script = scripts_dir / "plot_training.py"

    # Treina
    print("=" * 60)
    print("[FASE 1] Treinamento")
    print("=" * 60)
    train_args = [sys.executable, str(train_script)] + sys.argv[1:]
    result = subprocess.run(train_args)
    if result.returncode != 0:
        print("[ERRO] Treinamento falhou")
        sys.exit(1)

    # Encontra o log JSON
    # Tenta encontrar via --config para determinar save_dir
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break

    if config_path:
        # Le config para pegar save_dir
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        save_dir = cfg.get("save_dir", "checkpoints")
    else:
        save_dir = "checkpoints"

    log_path = Path(save_dir) / "training_log.json"
    if not log_path.exists():
        print(f"[WARN] Log nao encontrado em {log_path}, pulando graficos")
        sys.exit(0)

    # Gera graficos
    print("\n" + "=" * 60)
    print("[FASE 2] Graficos")
    print("=" * 60)

    # Extrai nome da escala do config
    label = Path(config_path).stem if config_path else "training"
    plot_dir = Path(save_dir) / "plots"

    plot_args = [
        sys.executable, str(plot_script),
        "--log", str(log_path),
        "--labels", label,
        "--output", str(plot_dir),
    ]
    result = subprocess.run(plot_args)
    if result.returncode != 0:
        print("[WARN] Geracao de graficos falhou")
    else:
        print(f"\n[OK] Graficos em {plot_dir}/")


if __name__ == "__main__":
    main()
