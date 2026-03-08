#!/bin/bash
# ============================================================================
# Auto-sync checkpoints from RunPod pod to local machine
#
# Uso: ./scripts/sync_checkpoints.sh [--once] [--interval SECONDS]
#
# Monitora checkpoints novos no pod e baixa automaticamente.
# Para quando o treino terminar (arquivo TRAINING_DONE aparece).
# ============================================================================

set -euo pipefail

# --- Configuracao do Pod ---
POD_HOST="38.80.152.148"
POD_PORT="31258"
POD_KEY="$HOME/.ssh/id_ed25519"
POD_USER="root"
POD_CKPT_DIR="/root/aletheion-ckpt/350m_4xh100"
POD_LOG="/root/aletheion-ckpt/350m_4xh100/cloud_train.log"

# --- Local ---
LOCAL_CKPT_DIR="$HOME/dev/ai/aletheion-llm-v2/checkpoints/350m_4xh100"
SYNC_INTERVAL="${2:-300}"  # default 5 minutos
ONE_SHOT=false

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --once) ONE_SHOT=true; shift ;;
        --interval) SYNC_INTERVAL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${POD_PORT} -i ${POD_KEY} ${POD_USER}@${POD_HOST}"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

mkdir -p "$LOCAL_CKPT_DIR"

sync_checkpoints() {
    log "Sincronizando checkpoints..."

    # Lista checkpoints remotos
    REMOTE_CKPTS=$($SSH_CMD "ls ${POD_CKPT_DIR}/step_*.pt 2>/dev/null" 2>/dev/null || echo "")

    for REMOTE in $REMOTE_CKPTS; do
        FNAME=$(basename "$REMOTE")
        LOCAL_FILE="${LOCAL_CKPT_DIR}/${FNAME}"
        if [ ! -f "$LOCAL_FILE" ]; then
            log "Baixando ${FNAME}..."
            scp -P ${POD_PORT} -i ${POD_KEY} -o StrictHostKeyChecking=no \
                "${POD_USER}@${POD_HOST}:${REMOTE}" "${LOCAL_FILE}" 2>&1 | tail -1
            log "${FNAME} baixado."
        fi
    done

    # Baixa logs
    scp -P ${POD_PORT} -i ${POD_KEY} -o StrictHostKeyChecking=no \
        "${POD_USER}@${POD_HOST}:${POD_LOG}" "${LOCAL_CKPT_DIR}/cloud_train.log" 2>/dev/null || true

    # Mostra checkpoints locais
    log "Checkpoints locais:"
    ls -lh "${LOCAL_CKPT_DIR}"/step_*.pt 2>/dev/null || echo "  (nenhum ainda)"
}

show_training_status() {
    LAST_LINE=$($SSH_CMD "tail -1 ${POD_LOG} 2>/dev/null" 2>/dev/null || echo "")
    if [ -n "$LAST_LINE" ]; then
        log "Status do treino:"
        echo "  $LAST_LINE" | tr '|' '\n' | grep -E 'step=|ce=|total=|tokens_per_sec=' | sed 's/^/  /'
    fi
}

check_training_done() {
    # Verifica se o processo torchrun ainda esta rodando
    RUNNING=$($SSH_CMD "pgrep -f 'torchrun.*train_distributed' >/dev/null 2>&1 && echo 'yes' || echo 'no'" 2>/dev/null || echo "unknown")
    echo "$RUNNING"
}

# --- Main loop ---
log "============================================"
log "  Aletheion Checkpoint Auto-Sync"
log "  Pod: ${POD_HOST}:${POD_PORT}"
log "  Intervalo: ${SYNC_INTERVAL}s"
log "============================================"

while true; do
    # Status do treino
    show_training_status

    # Sync checkpoints
    sync_checkpoints

    if [ "$ONE_SHOT" = true ]; then
        log "Modo --once: sincronizacao unica completa."
        break
    fi

    # Verifica se treino terminou
    TRAINING_STATUS=$(check_training_done)
    if [ "$TRAINING_STATUS" = "no" ]; then
        log "Treino FINALIZADO! Fazendo sync final..."
        sync_checkpoints
        log "============================================"
        log "  Todos os checkpoints baixados!"
        log "  Local: ${LOCAL_CKPT_DIR}/"
        log "============================================"
        break
    fi

    log "Proximo sync em ${SYNC_INTERVAL}s... (Ctrl+C para parar)"
    sleep "$SYNC_INTERVAL"
done
