#!/usr/bin/env bash
# ============================================================
# AletheionV2 - Treinamento Interativo
# Bash script para Linux
# ============================================================
#
# Uso:
#   ./train.sh                     # Setup completo interativo
#   ./train.sh --step setup        # So instala dependencias
#   ./train.sh --step data         # So prepara dados
#   ./train.sh --step train        # So treina
#   ./train.sh --test-data         # Usa TinyStories (teste rapido)
#   ./train.sh --resume path.pt    # Resume de checkpoint
#   ./train.sh --resume auto       # Auto-resume ultimo checkpoint
# ============================================================

set -euo pipefail

# -------------------------------------------------------
# Cores
# -------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERRO]${NC} $*"; }
header(){ echo -e "${CYAN}--- $* ---${NC}"; }

# -------------------------------------------------------
# Diretorio raiz do projeto
# -------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/scripts"
CONFIGS="$ROOT/configs/scaling"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"

# -------------------------------------------------------
# Parse argumentos
# -------------------------------------------------------
STEP="all"
TEST_DATA=false
RESUME=""
DATA_DIR=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --step)      STEP="$2"; shift 2 ;;
        --test-data) TEST_DATA=true; shift ;;
        --resume)    RESUME="$2"; shift 2 ;;
        --data-dir)  DATA_DIR="$2"; shift 2 ;;
        --config)    CONFIG="$2"; shift 2 ;;
        -h|--help)
            echo "Uso: ./train.sh [--step setup|data|train] [--test-data] [--resume path|auto] [--config path] [--data-dir path]"
            exit 0 ;;
        *) err "Argumento desconhecido: $1"; exit 1 ;;
    esac
done

# -------------------------------------------------------
# Banner
# -------------------------------------------------------
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  AletheionV2 - Treinamento Interativo${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# -------------------------------------------------------
# Configs disponiveis
# -------------------------------------------------------
declare -A CONFIG_DESC=(
    ["1m.yaml"]="~2M params     | <1 GB VRAM  | CPU/debug"
    ["10m.yaml"]="~13M params    | <1 GB VRAM  | CPU/GPU"
    ["50m.yaml"]="~42M params    | ~2 GB VRAM  | 1x GPU"
    ["125m.yaml"]="~110M params   | ~4 GB VRAM  | 1x GPU"
    ["350m.yaml"]="~354M params   | ~9 GB VRAM  | 1x GPU"
    ["350m_rtx4090.yaml"]="~354M params   | ~9 GB VRAM  | 1x RTX 4090 (otimizado)"
    ["1.3b.yaml"]="~1.3B params   | ~20 GB VRAM | 1x A100"
    ["7b.yaml"]="~6.6B params   | ~80 GB VRAM | 4x A100"
    ["13b.yaml"]="~13B params    | ~160 GB VRAM| 8x A100"
    ["30b.yaml"]="~30B params    | ~360 GB VRAM| 16x A100"
    ["70b.yaml"]="~70B params    | ~840 GB VRAM| 32x A100"
    ["162b.yaml"]="~162B params   | ~2 TB VRAM  | 64x H100"
    ["250b.yaml"]="~250B params   | ~3 TB VRAM  | 128x H100"
    ["400b.yaml"]="~400B params   | ~5 TB VRAM  | 256x H100"
    ["640b.yaml"]="~644B params   | ~8 TB VRAM  | 512x H100"
)

CONFIG_ORDER=(
    "1m.yaml"
    "10m.yaml"
    "50m.yaml"
    "125m.yaml"
    "350m.yaml"
    "350m_rtx4090.yaml"
    "1.3b.yaml"
    "7b.yaml"
    "13b.yaml"
    "30b.yaml"
    "70b.yaml"
    "162b.yaml"
    "250b.yaml"
    "400b.yaml"
    "640b.yaml"
)

# -------------------------------------------------------
# Datasets disponiveis
# -------------------------------------------------------
declare -A DATASET_DESC=(
    ["tinystories"]="TinyStories        | ~500M tokens  | Historias curtas, debug rapido"
    ["openwebtext"]="OpenWebText        | ~8B tokens    | Links populares do Reddit"
    ["wikipedia"]="Wikipedia          | ~6B tokens    | Enciclopedico"
    ["fineweb"]="FineWeb            | ~15T tokens   | Web crawl filtrado"
    ["fineweb-edu"]="FineWeb-Edu        | ~1.3T tokens  | Web educacional (recomendado)"
    ["slimpajama"]="SlimPajama         | ~627B tokens  | Mix de 7 fontes"
)

DATASET_ORDER=(
    "tinystories"
    "openwebtext"
    "wikipedia"
    "fineweb"
    "fineweb-edu"
    "slimpajama"
)

declare -A DATASET_SUBSETS=(
    ["fineweb"]="sample-10BT sample-100BT sample-350BT"
    ["fineweb-edu"]="sample-10BT sample-100BT sample-350BT"
    ["wikipedia"]="20220301.en 20220301.pt"
)

declare -A DATASET_MAX_TOKENS=(
    ["tinystories"]="500000000"
    ["openwebtext"]="8000000000"
    ["wikipedia"]="6000000000"
    ["fineweb"]="7000000000"
    ["fineweb-edu"]="7000000000"
    ["slimpajama"]="7000000000"
)

# -------------------------------------------------------
# Funcoes auxiliares
# -------------------------------------------------------
check_venv() {
    if [[ ! -f "$PYTHON" ]]; then
        warn "venv nao encontrado. Criando..."
        python3 -m venv "$VENV"
        "$VENV/bin/pip" install --upgrade pip -q
    fi
}

check_cuda() {
    "$PYTHON" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False"
}

show_gpu_info() {
    "$PYTHON" -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  GPU: {name} ({mem:.0f} GB VRAM)')
    print(f'  torch: {torch.__version__}')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print(f'  torch: {torch.__version__} (CPU only)')
" 2>/dev/null || warn "torch nao instalado"
}

select_config() {
    if [[ -n "$CONFIG" ]]; then
        # Config ja foi passada via argumento
        if [[ ! -f "$CONFIG" ]]; then
            # Tenta como nome de arquivo dentro de configs/scaling/
            if [[ -f "$CONFIGS/$CONFIG" ]]; then
                CONFIG="$CONFIGS/$CONFIG"
            else
                err "Config nao encontrada: $CONFIG"
                exit 1
            fi
        fi
        return
    fi

    echo ""
    echo -e "${BOLD}Escolha o modelo:${NC}"
    echo ""

    local i=1
    for cfg in "${CONFIG_ORDER[@]}"; do
        local marker=""
        if [[ "$cfg" == "350m_rtx4090.yaml" ]]; then
            marker=" ${GREEN}<-- recomendado RTX 4090${NC}"
        fi
        printf "  ${BOLD}%2d)${NC} %-25s %s%b\n" "$i" "$cfg" "${CONFIG_DESC[$cfg]}" "$marker"
        ((i++))
    done

    echo ""
    read -rp "Escolha [1-${#CONFIG_ORDER[@]}] (default: 6 = 350m_rtx4090): " choice
    choice="${choice:-6}"

    if [[ "$choice" -lt 1 || "$choice" -gt "${#CONFIG_ORDER[@]}" ]] 2>/dev/null; then
        err "Opcao invalida: $choice"
        exit 1
    fi

    local selected="${CONFIG_ORDER[$((choice-1))]}"
    CONFIG="$CONFIGS/$selected"
    info "Config selecionada: $selected"
}

select_dataset() {
    if $TEST_DATA; then
        DATASET="tinystories"
        SUBSET=""
        MAX_TOKENS="500000000"
        info "Modo teste: TinyStories (500M tokens)"
        return
    fi

    echo ""
    echo -e "${BOLD}Escolha o dataset:${NC}"
    echo ""

    local i=1
    for ds in "${DATASET_ORDER[@]}"; do
        local marker=""
        if [[ "$ds" == "fineweb-edu" ]]; then
            marker=" ${GREEN}<-- recomendado${NC}"
        fi
        printf "  ${BOLD}%d)${NC} %-18s %s%b\n" "$i" "$ds" "${DATASET_DESC[$ds]}" "$marker"
        ((i++))
    done

    echo ""
    read -rp "Escolha [1-${#DATASET_ORDER[@]}] (default: 5 = fineweb-edu): " choice
    choice="${choice:-5}"

    if [[ "$choice" -lt 1 || "$choice" -gt "${#DATASET_ORDER[@]}" ]] 2>/dev/null; then
        err "Opcao invalida: $choice"
        exit 1
    fi

    DATASET="${DATASET_ORDER[$((choice-1))]}"
    info "Dataset selecionado: $DATASET"

    # Subset
    SUBSET=""
    if [[ -n "${DATASET_SUBSETS[$DATASET]:-}" ]]; then
        local subsets=(${DATASET_SUBSETS[$DATASET]})
        echo ""
        echo -e "${BOLD}Escolha o subset:${NC}"
        local j=1
        for s in "${subsets[@]}"; do
            printf "  ${BOLD}%d)${NC} %s\n" "$j" "$s"
            ((j++))
        done
        echo ""
        read -rp "Escolha [1-${#subsets[@]}] (default: 1): " schoice
        schoice="${schoice:-1}"
        SUBSET="${subsets[$((schoice-1))]}"
        info "Subset: $SUBSET"
    fi

    # Max tokens
    MAX_TOKENS="${DATASET_MAX_TOKENS[$DATASET]:-7000000000}"
    echo ""
    read -rp "Max tokens (default: $MAX_TOKENS): " custom_tokens
    MAX_TOKENS="${custom_tokens:-$MAX_TOKENS}"
}

# -------------------------------------------------------
# STEP 1: Setup
# -------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "setup" ]]; then
    header "Step 1: Verificando ambiente"

    check_venv

    # Verifica torch
    if [[ "$(check_cuda)" == "True" ]]; then
        info "CUDA disponivel"
        show_gpu_info
    else
        warn "CUDA nao disponivel"
        show_gpu_info

        echo ""
        read -rp "Instalar torch com CUDA 12.8? [S/n]: " resp
        resp="${resp:-S}"
        if [[ "$resp" =~ ^[SsYy] ]]; then
            warn "Instalando torch com CUDA..."
            "$VENV/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
            "$VENV/bin/pip" install -e "$ROOT" -q 2>/dev/null
            if [[ "$(check_cuda)" == "True" ]]; then
                info "CUDA instalado com sucesso!"
                show_gpu_info
            else
                err "CUDA nao detectado apos instalacao."
                err "Verifique se os drivers NVIDIA estao atualizados."
                exit 1
            fi
        else
            warn "Continuando sem CUDA (treinamento sera CPU - MUITO LENTO)"
        fi
    fi

    # Dependencias de dados
    echo ""
    header "Step 2: Verificando dependencias"
    "$VENV/bin/pip" install tiktoken datasets -q 2>/dev/null
    "$VENV/bin/pip" install -e "$ROOT" -q 2>/dev/null
    info "Dependencias instaladas"
fi

if [[ "$STEP" == "setup" ]]; then
    echo ""
    info "Setup completo. Execute './train.sh --step data' para preparar dados."
    exit 0
fi

# -------------------------------------------------------
# STEP 2: Preparar dados
# -------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "data" ]]; then
    echo ""
    header "Step 3: Preparando dados"

    select_config
    select_dataset

    # Nome do diretorio de output baseado no config
    CONFIG_NAME="$(basename "$CONFIG" .yaml)"
    if $TEST_DATA; then
        TARGET_DIR="$ROOT/data/${CONFIG_NAME}_test"
    else
        TARGET_DIR="$ROOT/data/$CONFIG_NAME"
    fi

    # Verifica se dados ja existem
    if [[ -n "$DATA_DIR" ]]; then
        TARGET_DIR="$DATA_DIR"
        info "Usando data-dir fornecido: $DATA_DIR"
    elif [[ -f "$TARGET_DIR/metadata.json" ]]; then
        local_bins=$(find "$TARGET_DIR" -name "*.bin" -size +1M 2>/dev/null | wc -l)
        if [[ "$local_bins" -gt 0 ]]; then
            info "Dados encontrados em $TARGET_DIR ($local_bins shards)"
            echo ""
            read -rp "Usar dados existentes? [S/n]: " resp
            resp="${resp:-S}"
            if [[ "$resp" =~ ^[SsYy] ]]; then
                DATA_DIR="$TARGET_DIR"
            fi
        fi
    fi

    if [[ -z "$DATA_DIR" ]]; then
        echo ""
        echo -e "${YELLOW}[DATA]${NC} Dataset:    $DATASET"
        [[ -n "$SUBSET" ]] && echo -e "${YELLOW}[DATA]${NC} Subset:     $SUBSET"
        echo -e "${YELLOW}[DATA]${NC} Max tokens: $(printf "%'d" "$MAX_TOKENS")"
        echo -e "${YELLOW}[DATA]${NC} Output:     $TARGET_DIR"
        echo ""

        PREP_ARGS=(
            "$SCRIPTS/prepare_data.py"
            --dataset "$DATASET"
            --output "$TARGET_DIR"
            --max-tokens "$MAX_TOKENS"
        )
        [[ -n "$SUBSET" ]] && PREP_ARGS+=(--subset "$SUBSET")

        "$PYTHON" "${PREP_ARGS[@]}"

        if [[ $? -ne 0 ]]; then
            err "Falha na preparacao de dados"
            exit 1
        fi

        DATA_DIR="$TARGET_DIR"
    fi

    info "Dados prontos em: $DATA_DIR"
fi

if [[ "$STEP" == "data" ]]; then
    echo ""
    info "Dados preparados. Execute './train.sh --step train' para treinar."
    exit 0
fi

# -------------------------------------------------------
# STEP 3: Treinar
# -------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "train" ]]; then
    echo ""
    header "Step 4: Treinamento"

    # Seleciona config se ainda nao foi selecionada
    if [[ -z "$CONFIG" ]]; then
        select_config
    fi

    CONFIG_NAME="$(basename "$CONFIG" .yaml)"

    # Determina diretorio de dados
    if [[ -z "$DATA_DIR" ]]; then
        for candidate in \
            "$ROOT/data/$CONFIG_NAME" \
            "$ROOT/data/${CONFIG_NAME}_test" \
            "$ROOT/data/350m" \
            "$ROOT/data/350m_test"; do
            if [[ -f "$candidate/metadata.json" ]]; then
                local_bins=$(find "$candidate" -name "*.bin" -size +1M 2>/dev/null | wc -l)
                if [[ "$local_bins" -gt 0 ]]; then
                    DATA_DIR="$candidate"
                    break
                fi
            fi
        done
    fi

    if [[ -z "$DATA_DIR" ]]; then
        err "Nenhum dado encontrado. Execute './train.sh --step data' primeiro."
        exit 1
    fi

    echo ""
    echo -e "${CYAN}[TRAIN]${NC} Config: $CONFIG"
    echo -e "${CYAN}[TRAIN]${NC} Data:   $DATA_DIR"
    echo -e "${CYAN}[TRAIN]${NC} Output: checkpoints/$CONFIG_NAME/"
    echo ""

    TRAIN_ARGS=(
        "$SCRIPTS/train_distributed.py"
        --config "$CONFIG"
        --data-dir "$DATA_DIR"
    )

    # Resume
    if [[ -n "$RESUME" ]]; then
        TRAIN_ARGS+=(--resume "$RESUME")
        warn "Resumindo de: $RESUME"
    else
        # Auto-resume se existem checkpoints
        CKPT_DIR="$ROOT/checkpoints/$CONFIG_NAME"
        if ls "$CKPT_DIR"/step_*.pt 1>/dev/null 2>&1; then
            warn "Checkpoints anteriores encontrados. Usando auto-resume."
            TRAIN_ARGS+=(--resume auto)
        fi
    fi

    "$PYTHON" "${TRAIN_ARGS[@]}"
    TRAIN_EXIT=$?

    if [[ $TRAIN_EXIT -eq 0 ]]; then
        echo ""
        info "Treinamento completo!"
        info "Checkpoints em: checkpoints/$CONFIG_NAME/"

        # Gera plots se possivel
        LOG_FILE="$ROOT/checkpoints/$CONFIG_NAME/training_log.json"
        if [[ -f "$LOG_FILE" && -f "$SCRIPTS/plot_training.py" ]]; then
            echo ""
            warn "Gerando visualizacoes..."
            "$PYTHON" "$SCRIPTS/plot_training.py" --log "$LOG_FILE" --output "$ROOT/plots/$CONFIG_NAME" || true
        fi
    else
        echo ""
        err "Treinamento falhou (exit code: $TRAIN_EXIT)"
        exit $TRAIN_EXIT
    fi
fi
