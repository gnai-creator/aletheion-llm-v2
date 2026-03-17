#!/bin/bash
# ============================================================
# Fase 1: Full Mahalanobis - Modelo 1M - RTX 4090
#
# G constante SPD completo (espaco plano, off-diagonal).
# Hipotese: correlacoes entre eixos epistemicos melhoram ECE.
#
# Estimativa: ~30-60s no RTX 4090
#
# Uso:
#   cd aletheion-llm-v2
#   bash scripts/run_4phase_1m.sh
#
# Proxima fase (se resultados positivos):
#   git checkout real_geodesic
#   .venv/bin/python scripts/train_distributed.py \
#     --config configs/scaling/1m_real_geodesic_rtx4090.yaml \
#     --resume checkpoints/1m_full_mahalanobis/latest.pt
# ============================================================

set -e

PYTHON=".venv/bin/python"
TRAIN="scripts/train_distributed.py"
CONFIGS="configs/scaling"
LOG="logs/1m_full_mahalanobis_$(date +%Y%m%d_%H%M%S).log"

mkdir -p logs

echo "============================================================"
echo "  Full Mahalanobis (G constante, espaco plano)"
echo "  Config: $CONFIGS/1m_full_mahalanobis_rtx4090.yaml"
echo "  Log: $LOG"
echo "============================================================"

$PYTHON $TRAIN \
    --config $CONFIGS/1m_full_mahalanobis_rtx4090.yaml \
    2>&1 | tee "$LOG"

echo ""
echo "============================================================"
echo "  TREINO COMPLETO"
echo "  Checkpoint: checkpoints/1m_full_mahalanobis/"
echo "  Log: $LOG"
echo ""
echo "  Proximos passos:"
echo "    1. Avaliar ECE/Brier nos logs"
echo "    2. Se ECE < baseline -> correlacoes existem"
echo "    3. git checkout real_geodesic"
echo "    4. Rodar com --resume checkpoints/1m_full_mahalanobis/latest.pt"
echo "============================================================"
