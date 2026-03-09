#!/bin/bash
# Launch epistemic fine-tuning on 4x H200 SXM
set -euo pipefail

POD_HOST="213.181.104.59"
POD_PORT="16122"
POD_KEY="$HOME/.ssh/id_ed25519"
SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${POD_PORT} -i ${POD_KEY} root@${POD_HOST}"

REPO="/workspace/aletheion-llm-v2"
CKPT="/root/aletheion-ckpt/350m_epistemic_finetune/step_2000.pt"
DATA="/root/aletheion-data/350m_rtx4090"
CONFIG="configs/scaling/350m_epistemic_finetune_4xh200.yaml"
OUTDIR="/root/aletheion-ckpt/350m_epistemic_finetune"

echo "[1/4] Verificando GPUs..."
$SSH_CMD "nvidia-smi -L"

echo "[2/4] Clonando repo e instalando deps..."
$SSH_CMD "cd /workspace && (git clone https://github.com/gnai-creator/aletheion-llm-v2.git 2>/dev/null || cd aletheion-llm-v2 && git pull) && pip install -q tiktoken datasets"

echo "[3/4] Preparando dados..."
$SSH_CMD "mkdir -p ${OUTDIR} ${DATA} && cd ${REPO} && nohup python scripts/prepare_data.py --dataset fineweb-edu --subset sample-10BT --output ${DATA} > /tmp/prepare_data.log 2>&1 &"

echo "[4/4] Lancando fine-tuning..."
$SSH_CMD "cd ${REPO} && \
    nohup torchrun --nproc_per_node=4 \
        scripts/train_distributed.py \
        --config ${CONFIG} \
        --resume ${CKPT} \
        --data-dir ${DATA} \
        --override save_dir=${OUTDIR} \
    > ${OUTDIR}/finetune.log 2>&1 &

    sleep 2
    echo 'Lancado!'
"
