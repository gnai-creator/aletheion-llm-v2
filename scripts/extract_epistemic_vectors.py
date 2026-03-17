"""
Extracao de vetores epistemicos 5D do AletheionV2.

Coleta 5 escalares por token do output do modelo:
    [q1, q2, confidence, vi_magnitude, phi_total]

Salva como .npy para uso em voronoi_foliation.py.

Uso:
    .venv/bin/python scripts/extract_epistemic_vectors.py \
        --checkpoint checkpoints/350m_epistemic_finetune_v2/final.pt \
        --output-dir eval_results/foliation \
        --max-tokens 1000000
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configura logging padrao."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_model(
    checkpoint_path: str, device: str = "cuda"
) -> Tuple[AletheionV2Model, AletheionV2Config, int]:
    """Carrega modelo de um checkpoint.

    Args:
        checkpoint_path: Caminho para o checkpoint .pt
        device: Device alvo (cuda ou cpu)

    Returns:
        Tupla (model, config, step)
    """
    logger.info("[LOAD] Carregando %s...", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in ckpt:
        config = ckpt["config"]
        if isinstance(config, dict):
            config = AletheionV2Config(**config)
    else:
        config = AletheionV2Config()

    model = AletheionV2Model(config)

    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)

    model = model.to(device).eval()
    step = ckpt.get("global_step", -1)
    params = sum(p.numel() for p in model.parameters())
    logger.info("[LOAD] %s params, step=%s", f"{params:,}", step)
    return model, config, step


def prepare_tokens(
    config: AletheionV2Config,
    data_dir: Optional[str] = None,
    max_tokens: int = 1_000_000,
) -> np.ndarray:
    """Prepara tokens de avaliacao.

    Tenta WikiText-103 (OOD) primeiro, depois shards locais.

    Args:
        config: Configuracao do modelo
        data_dir: Diretorio opcional com shards de dados
        max_tokens: Numero maximo de tokens a extrair

    Returns:
        Array de token IDs (int64)
    """
    # Tenta WikiText-103 test (OOD)
    try:
        import tiktoken
        from datasets import load_dataset

        logger.info("[DATA] Baixando WikiText-103 test (OOD)...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join([x["text"] for x in ds if x["text"].strip()])
        enc = tiktoken.get_encoding("gpt2")
        tokens = np.array(enc.encode(text), dtype=np.int64)
        tokens = tokens[:max_tokens]
        logger.info("[DATA] WikiText-103: %s tokens", f"{len(tokens):,}")
        return tokens
    except Exception as e:
        logger.warning("[WARN] WikiText-103 indisponivel: %s", e)

    # Fallback: shards locais
    if data_dir is None:
        data_dir = str(
            Path(__file__).parent.parent / "data" / "350m_rtx4090"
        )

    from glob import glob

    shards = sorted(glob(os.path.join(data_dir, "train_shard_*.bin")))
    if not shards:
        logger.error("[ERROR] Sem dados disponiveis em %s", data_dir)
        sys.exit(1)

    # Usa ultimo shard como held-out
    eval_shard = shards[-1]
    logger.info("[DATA] Usando shard: %s", os.path.basename(eval_shard))
    tokens = np.memmap(eval_shard, dtype=np.uint16, mode="r")
    tokens = tokens[:max_tokens].astype(np.int64)
    logger.info("[DATA] %s tokens carregados", f"{len(tokens):,}")
    return tokens


@torch.no_grad()
def extract_vectors(
    model: AletheionV2Model,
    tokens: np.ndarray,
    seq_len: int = 1024,
    batch_size: int = 4,
    device: str = "cuda",
    max_seqs: int = 2000,
) -> Dict[str, np.ndarray]:
    """Extrai vetores epistemicos 5D do modelo.

    Para cada token, coleta:
        [q1, q2, confidence, vi_magnitude, phi_total]

    Args:
        model: Modelo AletheionV2 carregado
        tokens: Array de token IDs
        seq_len: Comprimento de cada sequencia
        batch_size: Tamanho do batch
        device: Device (cuda/cpu)
        max_seqs: Numero maximo de sequencias

    Returns:
        Dict com arrays numpy:
            - vectors: [N, 5] vetores epistemicos
            - token_ids: [N] token IDs correspondentes
            - drm_coords: [N, 5] coordenadas DRM brutas
    """
    model.eval()

    n_seqs = min(len(tokens) // (seq_len + 1), max_seqs)
    if n_seqs == 0:
        logger.error("[ERROR] Tokens insuficientes para formar sequencias")
        sys.exit(1)

    logger.info(
        "[EXTRACT] %d sequencias, batch_size=%d, seq_len=%d",
        n_seqs, batch_size, seq_len,
    )

    all_vectors = []
    all_token_ids = []
    all_drm_coords = []

    t0 = time.time()

    for i in range(0, n_seqs, batch_size):
        batch_seqs = min(batch_size, n_seqs - i)
        input_ids = []
        target_ids = []

        for j in range(batch_seqs):
            idx = (i + j) * (seq_len + 1)
            seq = tokens[idx: idx + seq_len + 1]
            input_ids.append(seq[:seq_len])
            target_ids.append(seq[1: seq_len + 1])

        input_ids_t = torch.tensor(
            np.array(input_ids), dtype=torch.long, device=device
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids_t, return_tomography=True)

        tomo = output.tomography
        if tomo is None:
            logger.warning(
                "[WARN] Tomografia nula no batch %d, pulando", i
            )
            continue

        # Extrair 5 escalares por token: [B, T, 1] -> [B*T]
        q1 = tomo.q1.cpu().float().reshape(-1, 1)           # [B*T, 1]
        q2 = tomo.q2.cpu().float().reshape(-1, 1)           # [B*T, 1]
        conf = tomo.confidence.cpu().float().reshape(-1, 1)  # [B*T, 1]
        phi = tomo.phi_total.cpu().float().reshape(-1, 1)    # [B*T, 1]

        # vi_magnitude = norm(vi_direction)
        vi_dir = tomo.vi_direction.cpu().float()  # [B, T, 5]
        vi_mag = vi_dir.norm(dim=-1, keepdim=True).reshape(-1, 1)  # [B*T, 1]

        # Vetor epistemico 5D
        vec = torch.cat([q1, q2, conf, vi_mag, phi], dim=-1)  # [B*T, 5]
        all_vectors.append(vec.numpy())

        # Token IDs (do input, nao do target)
        ids_flat = input_ids_t.cpu().reshape(-1).numpy()
        all_token_ids.append(ids_flat)

        # Coordenadas DRM brutas
        drm = tomo.drm_coords.cpu().float().reshape(-1, 5).numpy()
        all_drm_coords.append(drm)

        if (i // batch_size) % 100 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = (i * seq_len) / elapsed
            logger.info(
                "  [%d/%d] %.0f tok/s, elapsed=%.1fs",
                i, n_seqs, rate, elapsed,
            )

    elapsed = time.time() - t0
    logger.info("[EXTRACT] Concluido em %.1fs", elapsed)

    vectors = np.concatenate(all_vectors, axis=0)
    token_ids = np.concatenate(all_token_ids, axis=0)
    drm_coords = np.concatenate(all_drm_coords, axis=0)

    logger.info(
        "[EXTRACT] %s vetores extraidos, shape=%s",
        f"{vectors.shape[0]:,}", vectors.shape,
    )

    return {
        "vectors": vectors,
        "token_ids": token_ids,
        "drm_coords": drm_coords,
    }


def main() -> None:
    """Ponto de entrada principal para extracao de vetores epistemicos."""
    parser = argparse.ArgumentParser(
        description="Extracao de vetores epistemicos 5D do AletheionV2"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/350m_epistemic_finetune_v2/final.pt",
        help="Checkpoint do modelo",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results/foliation",
        help="Diretorio de saida para .npy",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-seqs", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--label",
        default="finetuned",
        help="Label para prefixo dos arquivos (finetuned, backbone)",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not torch.cuda.is_available() and args.device == "cuda":
        args.device = "cpu"
        logger.warning("[WARN] CUDA indisponivel, usando CPU")

    # Carrega modelo
    model, config, step = load_model(args.checkpoint, args.device)

    # Prepara tokens
    tokens = prepare_tokens(config, args.data_dir, args.max_tokens)

    # Extrai vetores
    results = extract_vectors(
        model, tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        max_seqs=args.max_seqs,
    )

    # Salva
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.label
    for name, arr in results.items():
        path = out_dir / f"{prefix}_{name}.npy"
        np.save(str(path), arr)
        logger.info("[SAVE] %s -> %s (shape=%s)", name, path, arr.shape)

    # Salva metadados
    import json
    meta = {
        "checkpoint": args.checkpoint,
        "step": step,
        "label": prefix,
        "n_vectors": int(results["vectors"].shape[0]),
        "vector_dims": ["q1", "q2", "confidence", "vi_magnitude", "phi_total"],
        "seq_len": args.seq_len,
        "max_tokens": args.max_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = out_dir / f"{prefix}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("[SAVE] Metadados -> %s", meta_path)

    logger.info("[OK] Extracao completa: %s vetores", f"{results['vectors'].shape[0]:,}")


if __name__ == "__main__":
    main()
