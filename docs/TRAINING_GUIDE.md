# Guia Completo de Treinamento: 1M a 162B

## Tabela de Escalas

| Escala | Params | d_model | Layers | Heads | Tokens Treino | GPU Minima | GPUs Recomendadas | VRAM Total | Tempo Estimado | Custo Cloud (USD) |
|--------|--------|---------|--------|-------|---------------|------------|-------------------|------------|----------------|-------------------|
| 1M | ~1M | 64 | 4 | 2 | 10M | Qualquer | 1x qualquer | 1 GB | 2 min | $0 (local) |
| 10M | ~10M | 256 | 6 | 4 | 200M | GTX 1060 | 1x qualquer | 2 GB | 15 min | $0 (local) |
| 50M | ~50M | 512 | 8 | 8 | 1B | GTX 1080 | 1x RTX 3090 | 4 GB | 2 horas | ~$2 |
| 125M | ~125M | 768 | 12 | 12 | 2.5B | RTX 3090 | 1x A100 40GB | 8 GB | 6 horas | ~$10 |
| 350M | ~350M | 1024 | 24 | 16 | 7B | A100 40GB | 1x A100 80GB | 16 GB | 1 dia | ~$50 |
| 1.3B | ~1.3B | 2048 | 24 | 16 | 26B | A100 80GB | 2x A100 80GB | 40 GB | 3 dias | ~$250 |
| 7B | ~7B | 4096 | 32 | 32 | 1T | 8x A100 80GB | 8x A100 80GB | 200 GB | 20 dias | ~$25,000 |
| 13B | ~13B | 5120 | 40 | 40 | 1T | 16x A100 | 16x A100 80GB | 400 GB | 30 dias | ~$50,000 |
| 30B | ~30B | 6656 | 60 | 52 | 1.4T | 32x A100 | 32x A100 80GB | 800 GB | 45 dias | ~$120,000 |
| 70B | ~70B | 8192 | 80 | 64 | 2T | 64x A100 | 64x H100 80GB | 2 TB | 60 dias | ~$500,000 |
| 162B | ~162B | 12288 | 96 | 96 | 3.5T | 128x H100 | 256x H100 80GB | 8 TB | 90 dias | ~$2,000,000 |

> Custos baseados em H100 80GB a ~$2.50/hora/GPU (AWS p5.48xlarge ou Lambda Cloud).
> Tempos assumem throughput otimizado (bf16, FSDP, gradient checkpointing).

---

## Passo a Passo Completo

### FASE 0: Ambiente

#### Hardware necessario por escala:
- **1M-50M**: Qualquer PC com GPU (ate GTX 1060)
- **125M-350M**: 1x RTX 3090/4090 ou A100
- **1.3B**: 2-4x A100 80GB
- **7B**: 8x A100 80GB (1 node)
- **13B-30B**: 2-4 nodes x 8x A100
- **70B+**: Cluster dedicado 8+ nodes

#### Software:
```bash
# Python 3.10+
conda create -n aletheion python=3.10
conda activate aletheion

# PyTorch (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Dependencias
cd aletheion-llm-v2
pip install -e ".[dev,data]"
pip install wandb  # Opcional: logging
```

#### Verificacao:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -m pytest tests/ -v  # Deve dar 85/85
```

---

### FASE 1: Preparacao de Dados

#### Datasets recomendados por escala:

| Escala | Dataset | Tokens | Tamanho em disco |
|--------|---------|--------|------------------|
| 1M-10M | tinystories | ~500M | ~1 GB |
| 50M-125M | openwebtext | ~8B | ~20 GB |
| 350M-1.3B | fineweb sample-10BT | ~10B | ~25 GB |
| 7B | fineweb sample-100BT | ~100B | ~250 GB |
| 13B-30B | fineweb sample-350BT | ~350B | ~800 GB |
| 70B+ | fineweb completo + slimpajama | 1-3T | 2-8 TB |

#### Download e tokenizacao:

```bash
# Exemplo: 125M (openwebtext, ~8B tokens)
python scripts/prepare_data.py \
    --dataset openwebtext \
    --output data/openwebtext \
    --shard-size 100000000

# Exemplo: 7B (fineweb 100B subset)
python scripts/prepare_data.py \
    --dataset fineweb \
    --subset sample-100BT \
    --output data/fineweb-100B \
    --shard-size 100000000

# Exemplo: com limite de tokens (para teste)
python scripts/prepare_data.py \
    --dataset fineweb \
    --subset sample-10BT \
    --output data/fineweb-10B \
    --max-tokens 1000000000
```

Apos a preparacao, o diretorio `data/` tera:
```
data/openwebtext/
  train_shard_00000.bin   # ~200 MB cada
  train_shard_00001.bin
  ...
  metadata.json           # Estatisticas
```

---

### FASE 2: Treinamento

#### 1M-50M (single GPU, local)
```bash
# 1M - smoke test (2 min)
python scripts/train_distributed.py --config configs/scaling/1m.yaml --data-dir data/tinystories

# 10M - validacao (15 min)
python scripts/train_distributed.py --config configs/scaling/10m.yaml --data-dir data/tinystories

# 50M - baseline (2 horas numa RTX 3090)
python scripts/train_distributed.py --config configs/scaling/50m.yaml --data-dir data/openwebtext
```

#### 125M-350M (single GPU potente)
```bash
# 125M (~6 horas numa A100)
python scripts/train_distributed.py \
    --config configs/scaling/125m.yaml \
    --data-dir data/openwebtext \
    --override wandb_project=aletheion

# 350M (~1 dia numa A100 80GB)
python scripts/train_distributed.py \
    --config configs/scaling/350m.yaml \
    --data-dir data/fineweb-10B \
    --override wandb_project=aletheion
```

#### 1.3B (multi-GPU DDP)
```bash
# 2 GPUs
torchrun --nproc_per_node=2 \
    scripts/train_distributed.py \
    --config configs/scaling/1.3b.yaml \
    --data-dir data/fineweb-10B
```

#### 7B (8 GPUs, FSDP)
```bash
# 8 GPUs no mesmo node
torchrun --nproc_per_node=8 \
    scripts/train_distributed.py \
    --config configs/scaling/7b.yaml \
    --data-dir data/fineweb-100B
```

#### 13B-30B (multi-node)
```bash
# Node 0 (master)
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 \
    scripts/train_distributed.py \
    --config configs/scaling/13b.yaml \
    --data-dir data/fineweb-350B

# Node 1
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=29500 \
    scripts/train_distributed.py \
    --config configs/scaling/13b.yaml \
    --data-dir data/fineweb-350B
```

#### 70B-162B (cluster dedicado)
```bash
# 70B: 8 nodes x 8 H100 = 64 GPUs
# Necessita: SLURM, DeepSpeed, ou similar
# Exemplo com torchrun:
torchrun --nnodes=8 --nproc_per_node=8 --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=29500 \
    scripts/train_distributed.py \
    --config configs/scaling/70b.yaml \
    --data-dir /shared/data/fineweb-full
```

#### Resume de checkpoint:
```bash
python scripts/train_distributed.py \
    --config configs/scaling/7b.yaml \
    --data-dir data/fineweb-100B \
    --resume checkpoints/step_5000.pt
```

---

### FASE 3: Avaliacao

```python
from aletheion_v2 import AletheionV2Config, AletheionV2Model
from aletheion_v2.inference.generator import Generator
from aletheion_v2.inference.dashboard_bridge import DashboardBridge
from aletheion_v2.tokenizer import AletheionTokenizer
import torch

# Carrega modelo
config = AletheionV2Config.from_yaml("configs/scaling/125m.yaml")
model = AletheionV2Model(config)
ckpt = torch.load("checkpoints/final.pt", weights_only=False)
model.load_state_dict(ckpt["model"])

# Tokenizer
tok = AletheionTokenizer("gpt2")

# Gera
prompt = "The meaning of life is"
input_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)

gen = Generator(model, max_new_tokens=100, top_k=50, top_p=0.9)
result = gen.generate(input_ids)

# Decodifica
output_text = tok.decode(result.token_ids)
print(f"Prompt: {prompt}")
print(f"Output: {output_text}")
print(f"Avg confidence: {result.avg_confidence:.3f}")
print(f"Avg phi: {result.avg_phi:.3f}")

# Dashboard
bridge = DashboardBridge()
snapshot = bridge.from_generation_result(result)
print(bridge.to_json(snapshot))
```

---

## Custos Detalhados por Escala

### Cloud (AWS/Lambda/Vast.ai)

| Escala | GPUs | Tipo | $/hora | Dias | Custo Total |
|--------|------|------|--------|------|-------------|
| 1M | 1 | T4 | $0.50 | 0.001 | **$0** |
| 10M | 1 | T4 | $0.50 | 0.01 | **$0.10** |
| 50M | 1 | A10G | $1.00 | 0.08 | **$2** |
| 125M | 1 | A100 40GB | $3.00 | 0.25 | **$18** |
| 350M | 1 | A100 80GB | $4.00 | 1 | **$96** |
| 1.3B | 2 | A100 80GB | $8.00 | 3 | **$576** |
| 7B | 8 | A100 80GB | $32.00 | 20 | **$15,360** |
| 13B | 16 | A100 80GB | $64.00 | 30 | **$46,080** |
| 30B | 32 | H100 80GB | $80.00 | 45 | **$86,400** |
| 70B | 64 | H100 80GB | $160.00 | 60 | **$230,400** |
| 162B | 256 | H100 80GB | $640.00 | 90 | **$1,382,400** |

> Vast.ai/Lambda Cloud podem ser 30-50% mais baratos que AWS.
> Spot instances podem reduzir custo em ~60% (mas precisa checkpoint robusto).

### Dados (armazenamento)

| Dataset | Tokens | Disco Raw | Disco Tokenizado |
|---------|--------|-----------|------------------|
| TinyStories | 500M | 1 GB | 1 GB |
| OpenWebText | 8B | 12 GB | 16 GB |
| FineWeb 10BT | 10B | 15 GB | 20 GB |
| FineWeb 100BT | 100B | 150 GB | 200 GB |
| FineWeb 350BT | 350B | 500 GB | 700 GB |
| FineWeb Full | 15T | 10 TB | 30 TB |
| SlimPajama | 627B | 800 GB | 1.2 TB |

### Eletricidade (se local)

| GPUs | Consumo | kWh/dia | R$/dia* |
|------|---------|---------|---------|
| 1x RTX 4090 | 450W | 10.8 | R$8 |
| 1x A100 | 400W | 9.6 | R$7 |
| 8x A100 | 3200W | 76.8 | R$56 |
| 8x H100 | 5600W | 134.4 | R$98 |

> *R$0.73/kWh (tarifa media BR residencial)

---

## Chinchilla Scaling Laws

Para treinamento otimo (Chinchilla), tokens = 20 * parametros:

| Params | Tokens Otimos | Nosso Config | Ratio |
|--------|---------------|--------------|-------|
| 1M | 20M | 10M | 10x |
| 10M | 200M | 200M | 20x |
| 125M | 2.5B | 2.5B | 20x |
| 1.3B | 26B | 26B | 20x |
| 7B | 140B | 1T | 143x |
| 70B | 1.4T | 2T | 29x |
| 162B | 3.2T | 3.5T | 22x |

> Configs maiores (7B+) treinam com mais tokens que Chinchilla recomenda
> porque o custo de inferencia justifica modelos "over-trained" (Llama approach).

---

## Recomendacao de Caminho

Se voce esta comecando do zero, siga esta sequencia:

1. **1M** - Valida que o pipeline funciona (2 min)
2. **10M** - Verifica convergencia basica (15 min)
3. **50M** - Primeiro modelo que gera texto coerente (2h)
4. **125M** - Baseline real, compara com GPT-2 small (6h)
5. **350M** - Melhoria significativa na qualidade (~1 dia)
6. **1.3B** - Instrucao-following basico emerge (~3 dias)
7. **7B** - Qualidade competitiva com modelos open-source (~20 dias)
8. **13B+** - Requer investimento serio em infra

Nao pule etapas. Cada escala valida que o pipeline funciona antes de investir mais.

---

## Troubleshooting

### OOM (Out of Memory)
- Reduza `batch_size`
- Aumente `gradient_accumulation_steps` proporcionalmente
- Habilite `gradient_checkpointing: true`
- Use `mixed_precision: bf16`
- Use `fsdp: true` para modelos > 1.3B

### Loss nao converge
- Verifique que dados estao tokenizados corretamente
- Reduza `learning_rate`
- Aumente `warmup_steps`
- Verifique `grad_clip` (default 1.0)

### Multi-node lento
- Verifique rede (precisa de InfiniBand ou RoCE para 70B+)
- Use NCCL backend com NCCL_IB_DISABLE=0
- Verifique que shards estao em storage local (nao NFS lento)
