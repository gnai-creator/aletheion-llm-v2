"""
Preparacao de dados: download, tokenizacao e sharding.

Datasets suportados:
- fineweb:      HuggingFaceFW/fineweb (15T tokens, web crawl filtrado)
- fineweb-edu:  HuggingFaceFW/fineweb-edu (1.3T tokens, educacional)
- slimpajama:   cerebras/SlimPajama-627B (627B tokens, mix)
- wikipedia:    wikipedia (6B tokens, enciclopedico)
- openwebtext:  Skylion007/openwebtext (8B tokens, Reddit links)
- tinystories:  roneneldan/TinyStories (pequeno, para debug)

Fluxo:
    1. Download streaming do HuggingFace Hub
    2. Tokeniza com tiktoken (GPT-2 encoding, 50257 vocab)
    3. Salva como shards .bin (numpy uint16)
    4. Cria arquivo de metadata .meta.json

Uso:
    python scripts/prepare_data.py --dataset fineweb --subset sample-10BT --output data/
    python scripts/prepare_data.py --dataset wikipedia --lang pt --output data/
    python scripts/prepare_data.py --dataset tinystories --output data/
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Mapa de datasets conhecidos
DATASETS = {
    "fineweb": {
        "hf_name": "HuggingFaceFW/fineweb",
        "text_field": "text",
        "subsets": ["sample-10BT", "sample-100BT", "sample-350BT"],
        "desc": "Web crawl filtrado. sample-10BT = ~10B tokens",
    },
    "fineweb-edu": {
        "hf_name": "HuggingFaceFW/fineweb-edu",
        "text_field": "text",
        "subsets": ["sample-10BT", "sample-100BT", "sample-350BT"],
        "desc": "Web educacional. sample-10BT = ~10B tokens",
    },
    "slimpajama": {
        "hf_name": "cerebras/SlimPajama-627B",
        "text_field": "text",
        "subsets": [],
        "desc": "Mix de 7 fontes, 627B tokens total",
    },
    "wikipedia": {
        "hf_name": "wikipedia",
        "text_field": "text",
        "subsets": ["20220301.en", "20220301.pt"],
        "desc": "Wikipedia. ~6B tokens EN, ~500M tokens PT",
    },
    "openwebtext": {
        "hf_name": "Skylion007/openwebtext",
        "text_field": "text",
        "subsets": [],
        "desc": "Links populares do Reddit, ~8B tokens",
    },
    "tinystories": {
        "hf_name": "roneneldan/TinyStories",
        "text_field": "text",
        "subsets": [],
        "desc": "Historias curtas simples, ~500M tokens. Bom para debug",
    },
}


def download_and_tokenize(
    dataset_key: str,
    subset: str,
    output_dir: str,
    tokenizer_name: str = "gpt2",
    shard_size: int = 100_000_000,  # 100M tokens por shard
    max_tokens: int = 0,  # 0 = sem limite
    split: str = "train",
):
    """Download streaming + tokenizacao + sharding.

    Args:
        dataset_key: Chave no DATASETS dict
        subset: Subset (ex: "sample-10BT")
        output_dir: Diretorio de saida
        tokenizer_name: Nome do tiktoken encoding
        shard_size: Tokens por shard
        max_tokens: Limite de tokens (0 = tudo)
        split: Split do dataset
    """
    from datasets import load_dataset
    from aletheion_v2.tokenizer import AletheionTokenizer

    ds_info = DATASETS[dataset_key]
    tokenizer = AletheionTokenizer(tokenizer_name)

    print(f"[PREP] Dataset: {dataset_key}")
    print(f"[PREP] HF: {ds_info['hf_name']}" + (f" ({subset})" if subset else ""))
    print(f"[PREP] Tokenizer: {tokenizer_name} (vocab={tokenizer.vocab_size})")
    print(f"[PREP] Shard size: {shard_size:,} tokens")
    print(f"[PREP] Output: {output_dir}")

    # Carrega em streaming
    kwargs = {"streaming": True, "split": split}
    if subset:
        kwargs["name"] = subset

    ds = load_dataset(ds_info["hf_name"], **kwargs)
    text_field = ds_info["text_field"]

    # Prepara diretorio
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Tokeniza e escreve shards
    shard_idx = 0
    buffer = []
    total_tokens = 0
    total_docs = 0

    dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32

    for example in ds:
        text = example.get(text_field, "")
        if not text or len(text.strip()) < 10:
            continue

        tokens = tokenizer.encode(text, add_eos=True)
        buffer.extend(tokens)
        total_docs += 1

        # Shard cheio?
        while len(buffer) >= shard_size:
            chunk = buffer[:shard_size]
            buffer = buffer[shard_size:]

            arr = np.array(chunk, dtype=dtype)
            shard_name = f"{split}_shard_{shard_idx:05d}.bin"
            arr.tofile(str(out_path / shard_name))

            total_tokens += len(chunk)
            shard_idx += 1

            print(
                f"  [SHARD {shard_idx}] {total_tokens:,} tokens, "
                f"{total_docs:,} docs"
            )

            # Limite de tokens
            if max_tokens > 0 and total_tokens >= max_tokens:
                break

        if max_tokens > 0 and total_tokens >= max_tokens:
            break

    # Ultimo shard parcial
    if buffer and (max_tokens == 0 or total_tokens < max_tokens):
        arr = np.array(buffer, dtype=dtype)
        shard_name = f"{split}_shard_{shard_idx:05d}.bin"
        arr.tofile(str(out_path / shard_name))
        total_tokens += len(buffer)
        shard_idx += 1

    # Metadata
    meta = {
        "dataset": dataset_key,
        "hf_name": ds_info["hf_name"],
        "subset": subset,
        "split": split,
        "tokenizer": tokenizer_name,
        "vocab_size": tokenizer.vocab_size,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "num_shards": shard_idx,
        "shard_size": shard_size,
        "dtype": "uint16" if dtype == np.uint16 else "uint32",
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[PREP] Completo!")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Docs:   {total_docs:,}")
    print(f"  Shards: {shard_idx}")
    print(f"  Output: {out_path}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Preparacao de dados AletheionV2")
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASETS.keys()),
        help="Dataset para preparar"
    )
    parser.add_argument("--subset", default="", help="Subset do dataset")
    parser.add_argument("--output", default="data", help="Diretorio de saida")
    parser.add_argument("--tokenizer", default="gpt2", help="Encoding tiktoken")
    parser.add_argument(
        "--shard-size", type=int, default=100_000_000,
        help="Tokens por shard (default: 100M)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=0,
        help="Limite de tokens (0 = sem limite)"
    )
    parser.add_argument("--split", default="train", help="Split do dataset")
    parser.add_argument("--list", action="store_true", help="Lista datasets")

    args = parser.parse_args()

    if args.list:
        print("\nDatasets disponiveis:")
        for key, info in DATASETS.items():
            print(f"  {key:15s} - {info['desc']}")
            if info["subsets"]:
                print(f"  {'':15s}   subsets: {', '.join(info['subsets'])}")
        return

    download_and_tokenize(
        args.dataset,
        args.subset,
        args.output,
        args.tokenizer,
        args.shard_size,
        args.max_tokens,
        args.split,
    )


if __name__ == "__main__":
    main()
