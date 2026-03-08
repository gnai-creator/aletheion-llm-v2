#!/bin/bash
# ============================================================================
# Deploy Aletheion-LLM-v2 Training on Cloud GPU (RunPod/Vast.ai)
#
# Uso:
#   1. Alugue uma instancia 4x H100 80GB no RunPod ou Vast.ai
#   2. SSH na instancia
#   3. Execute este script:
#      curl -sL <raw_github_url> | bash
#      OU
#      scp este script + dados para a instancia e execute
#
# Pre-requisitos na instancia:
#   - Ubuntu 22.04+ com CUDA 12.x
#   - Python 3.10+
#   - 4x H100 80GB (ou 4x A100 40/80GB)
#
# O script:
#   1. Clona o repo
#   2. Instala dependencias
#   3. Copia dados e checkpoint via rsync (ou baixa do HuggingFace)
#   4. Resume treino do ultimo checkpoint
#   5. Ao terminar, faz upload dos checkpoints finais
# ============================================================================

set -euo pipefail

# ============================================================================
# CONFIGURACAO - EDITAR ANTES DE RODAR
# ============================================================================

# IP/hostname da sua maquina local (para rsync dos dados)
LOCAL_HOST="SEU_IP_LOCAL"
LOCAL_USER="gnai-creator"
LOCAL_SSH_PORT=22

# Caminhos locais (na sua maquina)
LOCAL_PROJECT="/home/gnai-creator/dev/ai/aletheion-llm-v2"
LOCAL_DATA="${LOCAL_PROJECT}/data/350m_rtx4090"
LOCAL_CHECKPOINT_DIR="${LOCAL_PROJECT}/checkpoints/350m_rtx4090"

# Caminhos na instancia cloud
CLOUD_ROOT="/workspace/aletheion-llm-v2"
CLOUD_DATA="${CLOUD_ROOT}/data/350m_rtx4090"
CLOUD_CHECKPOINT_DIR="${CLOUD_ROOT}/checkpoints/350m_4xh100"

# Config
CONFIG="configs/scaling/350m_4xh100.yaml"
NUM_GPUS=4

# ============================================================================
# FUNCOES
# ============================================================================

log() { echo "[$(date '+%H:%M:%S')] $1"; }

check_gpus() {
    log "Verificando GPUs..."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log "GPUs detectadas: ${GPU_COUNT}"
    if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
        log "AVISO: Esperava ${NUM_GPUS} GPUs, encontrou ${GPU_COUNT}. Ajustando..."
        NUM_GPUS=$GPU_COUNT
    fi
}

setup_env() {
    log "=== Setup do ambiente ==="

    # Clona repo se nao existir
    if [ ! -d "$CLOUD_ROOT" ]; then
        log "Clonando repositorio..."
        git clone git@github.com:gnai-creator/aletheion-llm-v2.git "$CLOUD_ROOT"
    fi

    cd "$CLOUD_ROOT"

    # Cria venv
    if [ ! -d ".venv" ]; then
        log "Criando venv..."
        python3 -m venv .venv
    fi

    source .venv/bin/activate

    # Instala dependencias
    log "Instalando dependencias..."
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt
    pip install matplotlib  # Para plots no final

    log "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    log "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    log "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
}

transfer_data() {
    log "=== Transferindo dados ==="

    mkdir -p "$CLOUD_DATA"

    if [ "$LOCAL_HOST" = "SEU_IP_LOCAL" ]; then
        log "LOCAL_HOST nao configurado. Baixando dados do HuggingFace..."
        log "Executando prepare_data.py..."
        python scripts/prepare_data.py \
            --dataset fineweb-edu \
            --subset sample-10BT \
            --total-tokens 7000000000 \
            --output-dir "$CLOUD_DATA"
    else
        log "Transferindo dados de ${LOCAL_USER}@${LOCAL_HOST}..."
        rsync -avz --progress \
            -e "ssh -p ${LOCAL_SSH_PORT}" \
            "${LOCAL_USER}@${LOCAL_HOST}:${LOCAL_DATA}/" \
            "$CLOUD_DATA/"
    fi

    # Verifica
    SHARD_COUNT=$(ls "$CLOUD_DATA"/train_shard_*.bin 2>/dev/null | wc -l)
    log "Shards encontrados: ${SHARD_COUNT}"

    if [ "$SHARD_COUNT" -eq 0 ]; then
        log "ERRO: Nenhum shard de dados encontrado!"
        exit 1
    fi
}

transfer_checkpoint() {
    log "=== Transferindo checkpoint ==="

    mkdir -p "$CLOUD_CHECKPOINT_DIR"

    if [ "$LOCAL_HOST" = "SEU_IP_LOCAL" ]; then
        log "LOCAL_HOST nao configurado. Treino iniciara do zero."
        return
    fi

    # Pega o checkpoint mais recente
    LATEST=$(ssh -p ${LOCAL_SSH_PORT} ${LOCAL_USER}@${LOCAL_HOST} \
        "ls -t ${LOCAL_CHECKPOINT_DIR}/step_*.pt 2>/dev/null | head -1")

    if [ -z "$LATEST" ]; then
        log "Nenhum checkpoint encontrado. Treino iniciara do zero."
        return
    fi

    log "Transferindo checkpoint: ${LATEST}"
    rsync -avz --progress \
        -e "ssh -p ${LOCAL_SSH_PORT}" \
        "${LOCAL_USER}@${LOCAL_HOST}:${LATEST}" \
        "$CLOUD_CHECKPOINT_DIR/"

    CKPT_NAME=$(basename "$LATEST")
    log "Checkpoint transferido: ${CLOUD_CHECKPOINT_DIR}/${CKPT_NAME}"
}

run_training() {
    log "=== Iniciando treino: ${NUM_GPUS}x GPUs ==="

    cd "$CLOUD_ROOT"
    source .venv/bin/activate

    # Encontra ultimo checkpoint para resume
    RESUME_ARG=""
    LATEST_CKPT=$(ls -t ${CLOUD_CHECKPOINT_DIR}/step_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        log "Resumindo de: ${LATEST_CKPT}"
        RESUME_ARG="--resume ${LATEST_CKPT}"
    fi

    # Treina com torchrun (multi-GPU)
    log "Comando: torchrun --nproc_per_node=${NUM_GPUS} scripts/train_distributed.py"
    log "Config: ${CONFIG}"
    log "Data: ${CLOUD_DATA}"

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29500 \
        scripts/train_distributed.py \
        --config "${CONFIG}" \
        --data-dir "${CLOUD_DATA}" \
        ${RESUME_ARG} \
        2>&1 | tee "${CLOUD_CHECKPOINT_DIR}/cloud_train.log"

    log "=== Treino finalizado ==="
}

upload_results() {
    log "=== Resultados ==="

    cd "$CLOUD_ROOT"

    # Lista checkpoints gerados
    log "Checkpoints:"
    ls -lh ${CLOUD_CHECKPOINT_DIR}/step_*.pt 2>/dev/null

    # Gera plots
    if [ -f "scripts/plot_training.py" ]; then
        log "Gerando plots..."
        source .venv/bin/activate
        python scripts/plot_training.py \
            --log-dir "$CLOUD_CHECKPOINT_DIR" \
            --output-dir "$CLOUD_CHECKPOINT_DIR" 2>/dev/null || true
    fi

    if [ "$LOCAL_HOST" != "SEU_IP_LOCAL" ]; then
        log "Enviando resultados para maquina local..."
        rsync -avz --progress \
            -e "ssh -p ${LOCAL_SSH_PORT}" \
            "${CLOUD_CHECKPOINT_DIR}/" \
            "${LOCAL_USER}@${LOCAL_HOST}:${LOCAL_PROJECT}/checkpoints/350m_4xh100/"
        log "Checkpoints enviados para ${LOCAL_HOST}"
    else
        log "Configure LOCAL_HOST para upload automatico."
        log "Ou copie manualmente: scp -r ${CLOUD_CHECKPOINT_DIR} seu_pc:~/checkpoints/"
    fi
}

# ============================================================================
# MAIN
# ============================================================================

log "============================================"
log "  Aletheion-LLM-v2 Cloud Training Deploy"
log "  Config: 4x H100 80GB"
log "============================================"

check_gpus
setup_env
transfer_data
transfer_checkpoint
run_training
upload_results

log "============================================"
log "  TREINO COMPLETO!"
log "============================================"
