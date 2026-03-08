#!/bin/bash
# ============================================================================
# Launch epistemic fine-tuning on 4x H100 (RunPod)
#
# Uso: ./scripts/launch_finetune_4xh100.sh [CHECKPOINT_PATH]
#
# Roda apos o backbone terminar. Usa o melhor checkpoint como ponto de partida.
# ============================================================================

set -euo pipefail

# --- Pod config ---
POD_HOST="38.80.152.148"
POD_PORT="31258"
POD_KEY="$HOME/.ssh/id_ed25519"
POD_USER="root"
SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${POD_PORT} -i ${POD_KEY} ${POD_USER}@${POD_HOST}"

# --- Paths on pod ---
REPO="/workspace/aletheion-llm-v2"
CKPT_DIR="/root/aletheion-ckpt/350m_4xh100"
FINETUNE_DIR="/root/aletheion-ckpt/350m_epistemic_finetune"
DATA_DIR="/root/aletheion-data/350m_rtx4090"
CONFIG="configs/scaling/350m_epistemic_finetune_4xh100.yaml"

# --- Find best checkpoint ---
CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ]; then
    echo "[INFO] Buscando ultimo checkpoint no pod..."
    CHECKPOINT=$($SSH_CMD "ls -t ${CKPT_DIR}/step_*.pt 2>/dev/null | head -1")
    if [ -z "$CHECKPOINT" ]; then
        echo "[ERRO] Nenhum checkpoint encontrado em ${CKPT_DIR}/"
        exit 1
    fi
    echo "[INFO] Usando checkpoint: $CHECKPOINT"
fi

echo "============================================"
echo "  Aletheion Epistemic Fine-Tuning (4xH100)"
echo "  Checkpoint: $(basename $CHECKPOINT)"
echo "  Config: $CONFIG"
echo "============================================"

# --- 1. Sync code to pod ---
echo "[1/4] Sincronizando codigo..."
rsync -az --delete \
    -e "ssh -p ${POD_PORT} -i ${POD_KEY} -o StrictHostKeyChecking=no" \
    --exclude='.venv' --exclude='checkpoints' --exclude='data' --exclude='plots' --exclude='.git' --exclude='__pycache__' \
    /home/gnai-creator/dev/ai/aletheion-llm-v2/ \
    ${POD_USER}@${POD_HOST}:${REPO}/ 2>/dev/null || {
    echo "[WARN] rsync falhou, usando scp..."
    # Fallback: sync key files
    for f in src/aletheion_v2/config.py src/aletheion_v2/training/trainer_distributed.py configs/scaling/350m_epistemic_finetune_4xh100.yaml scripts/train_distributed.py; do
        scp -P ${POD_PORT} -i ${POD_KEY} -o StrictHostKeyChecking=no \
            "/home/gnai-creator/dev/ai/aletheion-llm-v2/$f" \
            "${POD_USER}@${POD_HOST}:${REPO}/$f" 2>/dev/null
    done
}

# --- 2. Create output dir ---
echo "[2/4] Preparando diretorios..."
$SSH_CMD "mkdir -p ${FINETUNE_DIR}"

# --- 3. Launch training ---
echo "[3/4] Lancando fine-tuning..."
$SSH_CMD "cd ${REPO} && \
    nohup torchrun --nproc_per_node=4 \
        scripts/train_distributed.py \
        --config ${CONFIG} \
        --resume ${CHECKPOINT} \
        --finetune \
        --data-dir ${DATA_DIR} \
        --override save_dir=${FINETUNE_DIR} \
    > ${FINETUNE_DIR}/finetune.log 2>&1 &

    sleep 2
    if pgrep -f 'torchrun.*train_distributed' > /dev/null; then
        echo '[OK] Fine-tuning iniciado!'
        echo 'PID:' \$(pgrep -f 'torchrun.*train_distributed' | head -1)
    else
        echo '[ERRO] Falha ao iniciar. Ultimas linhas do log:'
        tail -20 ${FINETUNE_DIR}/finetune.log
    fi
"

# --- 4. Monitor ---
echo "[4/4] Monitorando..."
echo ""
echo "Para acompanhar:"
echo "  ssh -p ${POD_PORT} -i ${POD_KEY} ${POD_USER}@${POD_HOST} 'tail -f ${FINETUNE_DIR}/finetune.log'"
echo ""
echo "Para baixar checkpoints:"
echo "  scp -P ${POD_PORT} -i ${POD_KEY} ${POD_USER}@${POD_HOST}:${FINETUNE_DIR}/step_*.pt checkpoints/350m_epistemic_finetune/"
