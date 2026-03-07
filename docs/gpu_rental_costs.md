# Aletheion-LLM-v2: Custo de Aluguel de GPUs por Configuracao

**Data:** 2026-03-07
**Autor:** Felipe Maya Muniz
**Base de precos:** Março 2026 (Lambda Labs, RunPod, Vast.ai, AWS, GCP)

---

## 1. Precos de Referencia por GPU (aluguel por hora)

| GPU | VRAM | Preco/h (spot/community) | Preco/h (on-demand) | Fonte |
|-----|------|--------------------------|---------------------|-------|
| T4 | 16 GB | US$ 0.10 - 0.20 | US$ 0.50 | Vast.ai, RunPod |
| RTX 4090 | 24 GB | US$ 0.30 - 0.50 | US$ 0.74 | Vast.ai, RunPod |
| A100 40GB | 40 GB | US$ 0.80 - 1.20 | US$ 1.89 | Lambda, RunPod |
| A100 80GB | 80 GB | US$ 1.20 - 1.80 | US$ 2.85 | Lambda, AWS |
| H100 80GB | 80 GB | US$ 2.00 - 2.85 | US$ 3.99 | Lambda, GCP |
| H100 SXM | 80 GB | US$ 2.50 - 3.50 | US$ 4.50 | Lambda, CoreWeave |

**Nota:** Precos spot/community podem variar 30-50%. On-demand e garantido.
Para clusters grandes (64+ GPUs), contratos reservados podem dar 30-50% de desconto.

---

## 2. Premissas de Calculo

### Throughput estimado por GPU (tokens/segundo, treino bf16)

| GPU | 350M | 1.3B | 7B | 13B | 70B | 162B+ |
|-----|------|------|----|-----|-----|-------|
| T4 | ~2K | - | - | - | - | - |
| RTX 4090 | ~10K | ~4K | - | - | - | - |
| A100 80GB | ~25K | ~12K | ~4K | ~2K | - | - |
| H100 80GB | ~45K | ~22K | ~8K | ~4K | ~1.5K | ~500-800 |

**Throughput por GPU em cluster** (com overhead de comunicacao FSDP/DDP):
- 2 GPUs: ~1.85x de 1 GPU
- 4 GPUs: ~3.5x de 1 GPU
- 8 GPUs: ~6.5x de 1 GPU
- 32 GPUs: ~24x de 1 GPU
- 64 GPUs: ~45x de 1 GPU
- 128 GPUs: ~85x de 1 GPU
- 256 GPUs: ~160x de 1 GPU
- 512 GPUs: ~300x de 1 GPU

**Eficiencia de scaling** cai com mais GPUs (~75% em 8, ~65% em 64, ~60% em 512).

### Formula

```
horas_treino = total_tokens / (throughput_por_gpu * num_gpus * eficiencia_scaling)
custo = horas_treino * num_gpus * preco_por_gpu_hora
```

---

## 3. Tabela Completa de Custos

### Modelos Pequenos (treinaveis em 1 GPU)

| Config | Params | Tokens | GPU Recomendada | Qtd | VRAM Total | Horas | Custo Spot | Custo On-Demand |
|--------|--------|--------|-----------------|-----|------------|-------|------------|-----------------|
| **1m** | ~2M | 10M | T4 / RTX 4090 | 1 | 16-24 GB | <0.01h (~30s) | **< US$ 0.01** | **< US$ 0.01** |
| **10m** | ~13M | 200M | T4 / RTX 4090 | 1 | 16-24 GB | ~0.1h (6 min) | **US$ 0.02** | **US$ 0.05** |
| **50m** | ~42M | 1B | RTX 4090 | 1 | 24 GB | ~1.5h | **US$ 0.50** | **US$ 1.10** |
| **125m** | ~110M | 2.5B | RTX 4090 | 1 | 24 GB | ~5h | **US$ 1.50** | **US$ 3.70** |
| **350m** | ~354M | 7B | RTX 4090 | 1 | 24 GB | ~8-12 dias | **US$ 60 - 100** | **US$ 140 - 210** |

### Modelos Medios (1 a 8 GPUs)

| Config | Params | Tokens | GPU Recomendada | Qtd | VRAM Total | Horas | Custo Spot | Custo On-Demand |
|--------|--------|--------|-----------------|-----|------------|-------|------------|-----------------|
| **1.3b** | ~1.3B | 26B | A100 80GB | 1 | 80 GB | ~600h (25 dias) | **US$ 720** | **US$ 1,700** |
| **7b** | ~6.6B | 1T | A100 80GB | 4 | 320 GB | ~2,900h (~120 dias) | **US$ 14,000** | **US$ 33,000** |
| | | | H100 80GB | 4 | 320 GB | ~1,600h (~67 dias) | **US$ 13,000** | **US$ 25,500** |

### Modelos Grandes (8 a 64 GPUs)

| Config | Params | Tokens | GPU Recomendada | Qtd | VRAM Total | Horas | Custo Spot | Custo On-Demand |
|--------|--------|--------|-----------------|-----|------------|-------|------------|-----------------|
| **13b** | ~13B | 1T | A100 80GB | 8 | 640 GB | ~3,800h (~160 dias) | **US$ 36,500** | **US$ 87,000** |
| | | | H100 80GB | 8 | 640 GB | ~1,900h (~80 dias) | **US$ 30,400** | **US$ 61,000** |
| **30b** | ~30B | 1.4T | H100 80GB | 16 | 1.3 TB | ~3,200h (~133 dias) | **US$ 128,000** | **US$ 230,000** |
| **70b** | ~70B | 2T | H100 80GB | 32 | 2.6 TB | ~5,100h (~213 dias) | **US$ 408,000** | **US$ 730,000** |

### Modelos Frontier (64 a 512 GPUs)

| Config | Params | Tokens | GPU Recomendada | Qtd | VRAM Total | Horas | Custo Spot | Custo On-Demand |
|--------|--------|--------|-----------------|-----|------------|-------|------------|-----------------|
| **162b** | ~162B | 3.5T | H100 SXM | 64 | 5.1 TB | ~4,800h (~200 dias) | **US$ 768,000** | **US$ 1,380,000** |
| **250b** | ~250B | 5T | H100 SXM | 128 | 10.2 TB | ~5,500h (~229 dias) | **US$ 1,760,000** | **US$ 3,170,000** |
| **400b** | ~400B | 8T | H100 SXM | 256 | 20.5 TB | ~6,800h (~283 dias) | **US$ 4,350,000** | **US$ 7,830,000** |
| **640b** | ~644B | 13T | H100 SXM | 512 | 40.9 TB | ~9,400h (~392 dias) | **US$ 12,000,000** | **US$ 21,500,000** |

---

## 4. Resumo Visual (Custo Spot)

```
Config      Params    Custo Spot        |  Barra relativa
--------    ------    ---------------   |  ---------------
1m          ~2M       < US$ 0.01        |
10m         ~13M      US$ 0.02          |
50m         ~42M      US$ 0.50          |
125m        ~110M     US$ 1.50          |  .
350m        ~354M     US$ 60-100        |  ..
1.3b        ~1.3B     US$ 720           |  ...
7b          ~6.6B     US$ 13,000        |  .......
13b         ~13B      US$ 30,400        |  ...........
30b         ~30B      US$ 128,000       |  ........................
70b         ~70B      US$ 408,000       |  .......................................
162b        ~162B     US$ 768,000       |  ..................................................
250b        ~250B     US$ 1,760,000     |  ...................................................................
400b        ~400B     US$ 4,350,000     |  ............................................................................
640b        ~644B     US$ 12,000,000    |  ......................................................................................
```

---

## 5. Conversao para Real (BRL)

Cambio estimado: US$ 1 = R$ 5.50 (Marco 2026)

| Config | Params | Custo Spot (USD) | Custo Spot (BRL) | Custo On-Demand (BRL) |
|--------|--------|------------------|------------------|-----------------------|
| 1m | ~2M | < US$ 0.01 | < R$ 0.06 | < R$ 0.06 |
| 10m | ~13M | US$ 0.02 | R$ 0.11 | R$ 0.28 |
| 50m | ~42M | US$ 0.50 | R$ 2.75 | R$ 6.05 |
| 125m | ~110M | US$ 1.50 | R$ 8.25 | R$ 20.35 |
| **350m** | **~354M** | **US$ 80** | **R$ 440** | **R$ 960** |
| 1.3b | ~1.3B | US$ 720 | R$ 3,960 | R$ 9,350 |
| **7b** | **~6.6B** | **US$ 13,000** | **R$ 71,500** | **R$ 140,250** |
| 13b | ~13B | US$ 30,400 | R$ 167,200 | R$ 335,500 |
| 30b | ~30B | US$ 128,000 | R$ 704,000 | R$ 1,265,000 |
| **70b** | **~70B** | **US$ 408,000** | **R$ 2,244,000** | **R$ 4,015,000** |
| 162b | ~162B | US$ 768,000 | R$ 4,224,000 | R$ 7,590,000 |
| 250b | ~250B | US$ 1,760,000 | R$ 9,680,000 | R$ 17,435,000 |
| 400b | ~400B | US$ 4,350,000 | R$ 23,925,000 | R$ 43,065,000 |
| **640b** | **~644B** | **US$ 12,000,000** | **R$ 66,000,000** | **R$ 118,250,000** |

---

## 6. Providers Recomendados por Escala

### 1 GPU (ate 350M)
- **Vast.ai**: Mais barato (community GPUs). RTX 4090 a ~US$ 0.30/h
- **RunPod**: Boa interface, serverless disponivel. RTX 4090 a ~US$ 0.50/h
- **Kaggle/Colab**: Gratis (T4/T4x2), limitado a 12h/sessao

### 1-8 GPUs (1.3B a 13B)
- **Lambda Labs**: A100/H100 on-demand, excelente network entre GPUs
- **RunPod**: Pods com multi-GPU, preco competitivo
- **Vast.ai**: Multi-GPU disponivel mas menos confiavel para runs longos

### 8-64 GPUs (13B a 162B)
- **Lambda Labs**: Clusters dedicados com InfiniBand
- **CoreWeave**: Especializado em GPU, bom pricing para H100
- **AWS p5 (H100)**: Caro mas confiavel. Spot instances podem economizar 60%
- **GCP a3 (H100)**: Committed use discounts de 1-3 anos

### 64-512 GPUs (250B a 640B)
- **CoreWeave**: Melhor preco para clusters grandes de H100
- **AWS/GCP/Azure**: Contratos reservados de 1-3 anos (30-50% desconto)
- **Lambda Labs Reserved**: Cluster dedicado com contrato
- **Nebius**: Novo player, precos agressivos em H100

**Nota:** Para 250B+ e necessario contrato direto com o provider.
Clusters de 128+ GPUs normalmente requerem InfiniBand/NVSwitch
e nao estao disponiveis em plataformas self-service.

---

## 7. Estrategia de Custo Otimizado

### Fase 1: Validacao (custo zero)
- Treinar 1m e 10m na RTX 4090 local (gratis)
- Validar arquitetura, loss convergence, tomografia

### Fase 2: Baseline funcional (R$ 0 - R$ 440)
- Treinar 350m na RTX 4090 local (~10 dias, custo zero de GPU)
- OU alugar 1x RTX 4090 spot por ~R$ 440 se quiser paralelizar

### Fase 3: Modelo competitivo (R$ 4K - R$ 72K)
- 1.3B: 1x A100 80GB, ~25 dias, ~R$ 4K spot
- 7B: 4x H100, ~67 dias, ~R$ 72K spot (primeiro modelo realmente util como LLM)

### Fase 4: Modelo frontier (R$ 167K+)
- Requer investimento ou grant
- 13B e o menor modelo que compete com GPT-4-mini
- 70B e o minimo para competir com Claude Sonnet

### Reducao de custos
- **Spot instances**: 50-70% mais baratas (risco de interrupcao)
- **Checkpointing frequente**: Salvar a cada 30min para sobreviver a interrupcoes spot
- **Committed use**: 1-3 anos de contrato = 30-50% desconto
- **Mixed cluster**: Treinar em A100 (mais barato) e fazer fine-tune em H100
- **Curriculo**: Treinar em dados menores primeiro, depois continuar com mais dados

---

## 8. Custo vs Retorno (Analise)

| Modelo | Custo Spot | Capacidade | ROI potencial |
|--------|-----------|------------|---------------|
| 350M | R$ 0 (local) | Co-processador epistemico para ATIC | Alto (custo zero) |
| 1.3B | R$ 4K | Backend para agents simples do ATIC | Alto |
| 7B | R$ 72K | LLM independente basico, substitui Ollama | Medio |
| 13B | R$ 167K | Compete com GPT-4-mini, backend completo ATIC | Medio |
| 70B | R$ 2.2M | Compete com Claude Sonnet, modelo comercial | Requer investimento |
| 640B | R$ 66M | Frontier, compete com GPT-4/Claude Opus | Requer Series A+ |

**Recomendacao:** Comecar com 350M (gratis), depois 1.3B (R$ 4K) para validar
a arquitetura em escala antes de investir em modelos maiores.

---

## Notas

- Precos baseados em cotacoes publicas de Marco 2026
- Throughput estimado assume treino otimizado (bf16, grad checkpointing, FSDP)
- Custos de armazenamento, rede e engenharia NAO estao incluidos
- Para modelos 70B+, adicionar ~10-20% para infra (armazenamento, networking)
- Custos de eletricidade para treino local (RTX 4090 ~350W): ~R$ 20-30 para 10 dias

---

(c) 2025-2026 Felipe Maya Muniz. All rights reserved.
