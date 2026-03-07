"""
Setup e lancamento de treinamento 350M no RTX 4090.

Etapas:
    1. Verifica torch CUDA
    2. Prepara dados (se necessario)
    3. Lanca treinamento

Uso:
    python scripts/setup_and_train.py                    # Setup completo
    python scripts/setup_and_train.py --skip-data        # Pula preparacao de dados
    python scripts/setup_and_train.py --data-only        # So prepara dados
    python scripts/setup_and_train.py --train-only       # So treina (assume tudo pronto)
    python scripts/setup_and_train.py --dry-run          # Mostra o que faria sem executar
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

# Raiz do projeto
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Config do treinamento
CONFIG_PATH = ROOT / "configs" / "scaling" / "350m_rtx4090.yaml"
DATA_DIR = ROOT / "data" / "350m"
DATASET = "fineweb-edu"
DATASET_SUBSET = "sample-10BT"
MAX_TOKENS = 7_000_000_000  # 7B tokens (Chinchilla-optimal para 350M)
SHARD_SIZE = 100_000_000  # 100M tokens por shard


def check_cuda() -> bool:
    """Verifica se torch com CUDA esta instalado."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"[OK] torch {torch.__version__} com CUDA")
            print(f"[OK] GPU: {name} ({mem:.0f} GB VRAM)")
            return True
        else:
            print(f"[WARN] torch {torch.__version__} sem CUDA")
            return False
    except ImportError:
        print("[WARN] torch nao instalado")
        return False


def install_cuda_torch() -> bool:
    """Instala torch com CUDA 12.4 (RTX 4090 / Ada Lovelace)."""
    print("\n[SETUP] Instalando torch com CUDA 12.4...")
    print("[SETUP] Isso pode demorar alguns minutos.\n")

    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[ERRO] Falha ao instalar torch CUDA.")
        print("[ERRO] Instale manualmente:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return False

    # Reinstala o projeto
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(ROOT)])
    return True


def check_data_deps() -> bool:
    """Verifica dependencias de dados."""
    ok = True
    try:
        import tiktoken
        print(f"[OK] tiktoken {tiktoken.__version__}")
    except ImportError:
        print("[WARN] tiktoken nao instalado")
        ok = False

    try:
        import datasets
        print(f"[OK] datasets {datasets.__version__}")
    except ImportError:
        print("[WARN] datasets nao instalado")
        ok = False

    if not ok:
        print("[SETUP] Instalando dependencias de dados...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "tiktoken>=0.5.0", "datasets>=2.14.0",
        ])
    return True


def check_data_ready() -> bool:
    """Verifica se dados ja estao preparados."""
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    total = meta.get("total_tokens", 0)
    n_shards = meta.get("num_shards", 0)

    # Verifica que os shards realmente existem e tem tamanho razoavel
    bin_files = list(DATA_DIR.glob("*.bin"))
    if not bin_files:
        return False

    # Verifica que nao sao placeholders (> 1MB cada)
    min_size = 1024 * 1024  # 1MB
    real_shards = [f for f in bin_files if f.stat().st_size > min_size]

    if not real_shards:
        print(f"[WARN] Shards em {DATA_DIR} parecem ser placeholders (< 1MB)")
        return False

    print(f"[OK] Dados: {total:,} tokens em {len(real_shards)} shards reais")
    return True


def prepare_data() -> bool:
    """Prepara dados para treinamento."""
    print(f"\n[DATA] Preparando {DATASET} ({DATASET_SUBSET})...")
    print(f"[DATA] Output: {DATA_DIR}")
    print(f"[DATA] Max tokens: {MAX_TOKENS:,}")
    print(f"[DATA] Isso pode demorar horas dependendo da conexao.\n")

    cmd = [
        sys.executable, str(ROOT / "scripts" / "prepare_data.py"),
        "--dataset", DATASET,
        "--subset", DATASET_SUBSET,
        "--output", str(DATA_DIR),
        "--max-tokens", str(MAX_TOKENS),
        "--shard-size", str(SHARD_SIZE),
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def prepare_small_data() -> bool:
    """Prepara dataset pequeno (TinyStories) para teste rapido."""
    small_dir = ROOT / "data" / "350m_test"
    print(f"\n[DATA] Preparando TinyStories para teste rapido...")
    print(f"[DATA] Output: {small_dir}")
    print(f"[DATA] ~500M tokens (suficiente para validar o pipeline)\n")

    cmd = [
        sys.executable, str(ROOT / "scripts" / "prepare_data.py"),
        "--dataset", "tinystories",
        "--output", str(small_dir),
        "--max-tokens", "500000000",  # 500M tokens
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def estimate_training_time():
    """Estima tempo de treinamento no RTX 4090."""
    try:
        import torch
        if not torch.cuda.is_available():
            return

        from aletheion_v2.config import AletheionV2Config
        config = AletheionV2Config.from_yaml(str(CONFIG_PATH))

        tokens_per_step = (
            config.batch_size * config.max_seq_len
            * config.gradient_accumulation_steps
        )
        total_steps = config.total_tokens // tokens_per_step

        # RTX 4090 throughput estimado: ~15K tokens/s com bf16 + compile
        tok_per_sec = 15000
        time_seconds = config.total_tokens / tok_per_sec
        time_hours = time_seconds / 3600
        time_days = time_hours / 24

        print(f"\n[EST] Estimativa de treinamento:")
        print(f"  Tokens por step: {tokens_per_step:,}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Throughput estimado: ~{tok_per_sec:,} tok/s")
        print(f"  Tempo estimado: ~{time_hours:.0f}h ({time_days:.1f} dias)")
    except Exception:
        pass


def run_training(data_dir: str = "") -> bool:
    """Lanca treinamento."""
    use_data = data_dir or str(DATA_DIR)

    # Verifica dados
    meta_path = Path(use_data) / "metadata.json"
    if not meta_path.exists():
        # Tenta dados de teste
        test_dir = ROOT / "data" / "350m_test"
        if (test_dir / "metadata.json").exists():
            use_data = str(test_dir)
            print(f"[TRAIN] Usando dados de teste: {use_data}")
        else:
            print("[ERRO] Nenhum dado encontrado. Execute com --data-only primeiro.")
            return False

    print(f"\n[TRAIN] Lancando treinamento 350M RTX 4090")
    print(f"[TRAIN] Config: {CONFIG_PATH}")
    print(f"[TRAIN] Data: {use_data}")
    print(f"[TRAIN] Checkpoints: checkpoints/350m_rtx4090/\n")

    cmd = [
        sys.executable, str(ROOT / "scripts" / "train_distributed.py"),
        "--config", str(CONFIG_PATH),
        "--data-dir", use_data,
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Setup e treinamento 350M RTX 4090"
    )
    parser.add_argument("--skip-data", action="store_true",
                        help="Pula preparacao de dados")
    parser.add_argument("--data-only", action="store_true",
                        help="So prepara dados, nao treina")
    parser.add_argument("--train-only", action="store_true",
                        help="So treina, assume tudo pronto")
    parser.add_argument("--test-data", action="store_true",
                        help="Prepara TinyStories (teste rapido)")
    parser.add_argument("--data-dir", default="",
                        help="Override do diretorio de dados")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra o que faria sem executar")
    args = parser.parse_args()

    print("=" * 60)
    print("  AletheionV2 - Treinamento 350M RTX 4090")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY-RUN] Verificando pre-requisitos...\n")

    # === STEP 1: Verificar CUDA ===
    print("\n--- Step 1: Verificando CUDA ---")
    if not args.train_only or not args.dry_run:
        has_cuda = check_cuda()
        if not has_cuda and not args.dry_run:
            print("\n[SETUP] torch CUDA nao encontrado.")
            resp = input("Instalar torch com CUDA 12.4? [S/n] ").strip().lower()
            if resp in ("", "s", "y", "sim", "yes"):
                if not install_cuda_torch():
                    return
                # Re-verifica
                if not check_cuda():
                    print("[ERRO] CUDA ainda nao disponivel apos instalacao.")
                    print("[ERRO] Verifique se os drivers NVIDIA estao instalados.")
                    return
            else:
                print("[INFO] Continuando sem CUDA (treinamento sera em CPU - LENTO)")

    # === STEP 2: Dependencias de dados ===
    if not args.train_only:
        print("\n--- Step 2: Verificando dependencias ---")
        check_data_deps()

    # === STEP 3: Preparar dados ===
    if not args.train_only and not args.skip_data:
        print("\n--- Step 3: Verificando dados ---")

        if args.test_data:
            if not args.dry_run:
                prepare_small_data()
        elif not check_data_ready():
            if args.dry_run:
                print(f"[DRY-RUN] Prepararia dados: {DATASET}/{DATASET_SUBSET}")
                print(f"[DRY-RUN] Output: {DATA_DIR}")
                print(f"[DRY-RUN] Tokens: {MAX_TOKENS:,}")
            else:
                print(f"\n[DATA] Dados nao encontrados em {DATA_DIR}")
                print(f"[DATA] Opcoes:")
                print(f"  1. Preparar {DATASET}/{DATASET_SUBSET} (7B tokens, ~horas)")
                print(f"  2. Preparar TinyStories (500M tokens, ~minutos)")
                print(f"  3. Pular (voce prepara depois)")
                resp = input("\nEscolha [1/2/3]: ").strip()
                if resp == "1":
                    if not prepare_data():
                        print("[ERRO] Falha na preparacao de dados")
                        return
                elif resp == "2":
                    if not prepare_small_data():
                        print("[ERRO] Falha na preparacao de dados")
                        return
                else:
                    print("[INFO] Pulando preparacao de dados")
        else:
            print("[OK] Dados ja preparados")

    if args.data_only:
        print("\n[DONE] Dados preparados. Execute sem --data-only para treinar.")
        return

    # === STEP 4: Estimativa ===
    estimate_training_time()

    # === STEP 5: Treinar ===
    print("\n--- Step 4: Treinamento ---")
    if args.dry_run:
        print(f"[DRY-RUN] Executaria treinamento com:")
        print(f"  Config: {CONFIG_PATH}")
        print(f"  Data: {args.data_dir or DATA_DIR}")
        print(f"\nComando:")
        print(f"  python scripts/train_distributed.py \\")
        print(f"    --config {CONFIG_PATH} \\")
        print(f"    --data-dir {args.data_dir or DATA_DIR}")
        return

    run_training(args.data_dir)


if __name__ == "__main__":
    main()
