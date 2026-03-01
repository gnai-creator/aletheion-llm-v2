#!/bin/bash
# Avaliacao do AletheionV2
# Uso: bash scripts/eval.sh [checkpoint_path]

CKPT=${1:-checkpoints/latest.pt}
python -m aletheion_v2.inference.generator --checkpoint "$CKPT"
