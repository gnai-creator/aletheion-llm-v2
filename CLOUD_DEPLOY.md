# Cloud Deploy - Aletheion-LLM-v2

## Passo a passo para treinar em 4x H100 (RunPod)

### 1. Criar pod no RunPod

1. Acesse [runpod.io](https://runpod.io)
2. **GPU Pod** → **Deploy**
3. Selecione: **4x H100 80GB SXM** (ou 4x H100 PCIe)
4. Template: **RunPod Pytorch 2.4** (CUDA 12.4)
5. Disco: **50GB** container + **100GB** volume (para dados)
6. Clique **Deploy**
7. Quando estiver ready, copie o SSH command

### 2. Configurar o script

Edite `scripts/deploy_cloud.sh` e preencha:

```bash
LOCAL_HOST="SEU_IP_PUBLICO"     # ip público da sua máquina (ou ngrok)
LOCAL_USER="gnai-creator"
LOCAL_SSH_PORT=22
```

Se não tiver IP público, use uma das alternativas:
- **ngrok**: `ngrok tcp 22` → use o host/port do ngrok
- **Upload manual**: copie dados com scp (ver abaixo)

### 3. Upload manual (se não tiver IP público)

```bash
# Na sua máquina, comprima os dados:
cd ~/dev/ai/aletheion-llm-v2
tar czf /tmp/aletheion-data.tar.gz data/350m_rtx4090/

# Copie para o pod (substitua XXXXX pelo pod ID):
scp -P 22XXX /tmp/aletheion-data.tar.gz root@ssh.runpod.io:/workspace/

# Copie o checkpoint mais recente:
scp -P 22XXX checkpoints/350m_rtx4090/step_2000.pt root@ssh.runpod.io:/workspace/

# No pod:
cd /workspace
git clone git@github.com:gnai-creator/aletheion-llm-v2.git
cd aletheion-llm-v2
tar xzf /workspace/aletheion-data.tar.gz
mkdir -p checkpoints/350m_4xh100
cp /workspace/step_2000.pt checkpoints/350m_4xh100/
```

### 4. Rodar o treino

```bash
# No pod via SSH:
cd /workspace/aletheion-llm-v2
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install matplotlib

# Treinar (4 GPUs, resume do checkpoint)
torchrun --nproc_per_node=4 --master_port=29500 \
    scripts/train_distributed.py \
    --config configs/scaling/350m_4xh100.yaml \
    --data-dir data/350m_rtx4090 \
    --resume checkpoints/350m_4xh100/step_2000.pt \
    2>&1 | tee checkpoints/350m_4xh100/cloud_train.log
```

### 5. Monitorar

```bash
# Em outro terminal SSH:
tail -f checkpoints/350m_4xh100/cloud_train.log

# Ou ver ultimo log:
tail -1 checkpoints/350m_4xh100/train.log | tr '|' '\n' | grep -E 'step=|ce=|total=|tokens_per_sec='
```

### 6. Baixar resultados

```bash
# Na sua máquina:
scp -P 22XXX -r root@ssh.runpod.io:/workspace/aletheion-llm-v2/checkpoints/350m_4xh100/ \
    ~/dev/ai/aletheion-llm-v2/checkpoints/350m_4xh100/
```

### 7. Desligar o pod

Depois de baixar os checkpoints, **termine o pod** no RunPod para parar a cobrança.

## Tempo estimado

| Fase | Tempo |
|------|-------|
| Setup + upload dados (14GB) | ~15-30 min |
| Treino (~104K steps restantes) | ~12-13 horas |
| Download checkpoints | ~10 min |
| **Total** | **~13-14 horas** |

## Custo estimado

| Cenário | $/hr | Total |
|---------|------|-------|
| 4x H100 spot (RunPod) | ~$8-10 | **$105-130** |
| 4x H100 on-demand | ~$14 | **$185-195** |

## Alternativa: Vast.ai

Vast.ai costuma ser mais barato. Mesmos passos, mas:
1. Acesse [vast.ai](https://vast.ai)
2. Filtre: 4x H100, PyTorch, CUDA 12+
3. Alugue a instância mais barata
4. SSH e siga os passos 3-7 acima

## Config differences (4xH100 vs RTX 4090)

| Param | RTX 4090 | 4x H100 |
|-------|----------|---------|
| batch_size | 4 | 16 |
| grad_accum | 16 | 1 |
| effective_batch | 64 | 64 |
| grad_checkpointing | true | false |
| compile_model | false | true |
| num_workers | 4 | 8 |
| prefetch_factor | 2 | 4 |

O effective batch (64) é mantido idêntico para não alterar a dinâmica do treino.
