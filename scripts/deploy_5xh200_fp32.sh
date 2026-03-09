#!/bin/bash
# ============================================================================
# Deploy Aletheion-LLM-v2 - 5x H200 SXM fp32 puro
#
# Uso na maquina LOCAL:
#   ./scripts/deploy_5xh200_fp32.sh <SSH_DESTINO>
#
# Exemplo:
#   ./scripts/deploy_5xh200_fp32.sh "root@38.80.152.148 -p 31258"
#   ./scripts/deploy_5xh200_fp32.sh "user@ssh.runpod.io -p 12345"
#
# O script:
#   1. Faz upload do codigo + checkpoint via rsync/scp
#   2. Instala dependencias no pod
#   3. Lanca treino fp32 com torchrun 5 GPUs
#   4. Mostra comando para monitorar
# ============================================================================

set -euo pipefail

# ============================================================================
# CONFIGURACAO
# ============================================================================

LOCAL_PROJECT="/home/gnai-creator/dev/ai/aletheion-llm-v2"
LOCAL_CHECKPOINT="${LOCAL_PROJECT}/checkpoints/350m_epistemic_finetune/step_11000.pt"
SSH_KEY="$HOME/.ssh/id_ed25519"

CONFIG="configs/scaling/350m_epistemic_finetune_5xh200_fp32.yaml"
NUM_GPUS=5

CLOUD_ROOT="/root/aletheion-llm-v2"
CLOUD_CKPT_DIR="${CLOUD_ROOT}/checkpoints/350m_epistemic_finetune"
CLOUD_LOG="${CLOUD_CKPT_DIR}/cloud_train.log"

# ============================================================================
# PARSE ARGS
# ============================================================================

if [ $# -lt 1 ]; then
    echo "Uso: $0 <SSH_DESTINO>"
    echo "  Ex: $0 \"root@1.2.3.4 -p 12345\""
    exit 1
fi

SSH_DEST="$1"
SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i ${SSH_KEY} ${SSH_DEST}"
SCP_BASE="scp -o StrictHostKeyChecking=no -i ${SSH_KEY}"

# Extrai host e porta para rsync
SSH_HOST=$(echo "$SSH_DEST" | awk '{print $1}')
SSH_PORT=$(echo "$SSH_DEST" | grep -oP '(?<=-p\s)\d+' || echo "22")

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# ============================================================================
# 1. VERIFICAR CONEXAO E GPUs
# ============================================================================

log "=== Conectando ao pod ==="
${SSH_CMD} "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
GPU_COUNT=$(${SSH_CMD} "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
log "GPUs detectadas: ${GPU_COUNT}"

if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    log "AVISO: Esperava ${NUM_GPUS} GPUs, encontrou ${GPU_COUNT}. Ajustando..."
    NUM_GPUS=$GPU_COUNT
fi

# ============================================================================
# 2. UPLOAD CODIGO
# ============================================================================

log "=== Upload do codigo ==="

# Cria diretorio remoto
${SSH_CMD} "mkdir -p ${CLOUD_ROOT}"

# Rsync do codigo (exclui dados, checkpoints, venv, caches)
rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=no -i ${SSH_KEY} -p ${SSH_PORT}" \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='data/' \
    --exclude='checkpoints/' \
    --exclude='wandb/' \
    --exclude='.mypy_cache/' \
    --exclude='.pytest_cache/' \
    "${LOCAL_PROJECT}/" \
    "${SSH_HOST}:${CLOUD_ROOT}/"

log "Codigo enviado."

# ============================================================================
# 3. UPLOAD CHECKPOINT
# ============================================================================

log "=== Upload do checkpoint step_11000 (4GB) ==="

${SSH_CMD} "mkdir -p ${CLOUD_CKPT_DIR}"

rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=no -i ${SSH_KEY} -p ${SSH_PORT}" \
    "${LOCAL_CHECKPOINT}" \
    "${SSH_HOST}:${CLOUD_CKPT_DIR}/"

log "Checkpoint enviado."

# ============================================================================
# 4. UPLOAD DADOS
# ============================================================================

log "=== Upload dos dados ==="

LOCAL_DATA="${LOCAL_PROJECT}/data/350m_rtx4090"
CLOUD_DATA="${CLOUD_ROOT}/data/350m_rtx4090"

${SSH_CMD} "mkdir -p ${CLOUD_DATA}"

rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=no -i ${SSH_KEY} -p ${SSH_PORT}" \
    "${LOCAL_DATA}/" \
    "${SSH_HOST}:${CLOUD_DATA}/"

log "Dados enviados."

# ============================================================================
# 5. INSTALAR DEPENDENCIAS
# ============================================================================

log "=== Instalando dependencias no pod ==="

${SSH_CMD} << 'REMOTE_SETUP'
set -e
cd /root/aletheion-llm-v2

# Usa python do sistema (pods RunPod ja tem torch+CUDA)
pip install --quiet pyyaml tiktoken numpy 2>/dev/null || true

# Verifica torch e CUDA
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1e9:.0f}GB)')
"

# Instala o pacote em modo editable
pip install -e . --quiet 2>/dev/null || true

echo "[OK] Setup completo"
REMOTE_SETUP

log "Dependencias instaladas."

# ============================================================================
# 6. LANCAR TREINO
# ============================================================================

log "=== Lancando treino fp32: ${NUM_GPUS}x GPUs ==="

${SSH_CMD} << REMOTE_TRAIN
set -e
cd /root/aletheion-llm-v2
mkdir -p ${CLOUD_CKPT_DIR}

# Mata processos anteriores se houver
pkill -f torchrun 2>/dev/null || true
pkill -f train_distributed 2>/dev/null || true
sleep 2

# Lanca treino em background com nohup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup torchrun \\
    --nproc_per_node=${NUM_GPUS} \\
    --master_port=29500 \\
    scripts/train_distributed.py \\
    --config ${CONFIG} \\
    --resume ${CLOUD_CKPT_DIR}/step_11000.pt \\
    > ${CLOUD_LOG} 2>&1 &

echo "PID: \$!"
echo "[OK] Treino lancado em background"
REMOTE_TRAIN

log "=== TREINO LANCADO ==="

# ============================================================================
# 7. MOSTRAR COMANDOS UTEIS
# ============================================================================

echo ""
echo "============================================"
echo "  DEPLOY COMPLETO - 5x H200 fp32"
echo "============================================"
echo ""
echo "Monitorar treino:"
echo "  ${SSH_CMD} \"tail -f ${CLOUD_LOG}\""
echo ""
echo "Ver status rapido:"
echo "  ${SSH_CMD} \"tail -5 ${CLOUD_LOG}; echo '---'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader\""
echo ""
echo "Verificar se esta rodando:"
echo "  ${SSH_CMD} \"pgrep -f torchrun && echo RUNNING || echo STOPPED\""
echo ""
echo "Download checkpoints quando terminar:"
echo "  rsync -avz --progress -e 'ssh -o StrictHostKeyChecking=no -i ${SSH_KEY} -p ${SSH_PORT}' ${SSH_HOST}:${CLOUD_CKPT_DIR}/ ${LOCAL_PROJECT}/checkpoints/350m_epistemic_finetune/"
echo ""
