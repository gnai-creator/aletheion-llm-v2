# Aletheion LLM v2

LLM decoder-only com sistema epistemico integrado como `nn.Module` treinaveis.
Cada token produz **tomografia epistemica** completa no manifold Riemanniano 5D.

**354M params** | **261 testes** | **17 configs (1M-640B)** | **Continual Learning (EWC + Replay)**

## Arquitetura

```
input_ids -> Embeddings (RoPE) -> TransformerBlock x N -> ln_final
                                                           |
                                              +------------+-----------+
                                              |                        |
                                          lm_head                EpistemicHead
                                              |                        |
                                          logits        EpistemicTomography (30+ campos)
                                                               |
                                          +----+----+----+----+----+----+----+
                                          |    |    |    |    |    |    |    |
                                         DRM  MAD  VI  MPC Eidos Phil SelfM ...
```

### Modulos Neurais (11 heads)

| Modulo | Params | Descricao |
|--------|--------|-----------|
| DRM Manifold | ~50K | Coordenadas 5D + tensor metrico + geodesicas |
| MAD Confidence | ~2K | Confianca Bayesiana com tau aprendivel |
| VI (Intentionality) | ~5K | Saude do manifold phi(M) + correcao |
| MPC Navigator | ~1K | Controle preditivo com beam search |
| EidosDecay | ~368 | Balanceamento de eixos 5D (decay/reinforce) |
| Filosofia3 | ~373 | Deteccao conflito phi-psi |
| SelfModel | ~982 | Humor, curiosidade, energia, drives |
| Grounding | ~99K | Classificacao tarefas + deteccao ambiguidade |
| PlasticityGate | ~12K | Gate de plasticidade com deplecao |
| MPL/Frontier | ~129 | Exploracao de fronteira no manifold |
| MOPsi | ~25K | Estado humano 5D + mediacao phi-psi |
| CausalState | ~26K | Condicionamento causal + policy binding |
| Metacognitive | ~200K | Auto-avaliacao contrastiva dual |

### EpistemicTomography (por token)

| Campo | Shape | Descricao |
|-------|-------|-----------|
| q1 | [B,T,1] | Incerteza aleatoria |
| q2 | [B,T,1] | Incerteza epistemica |
| confidence | [B,T,1] | MAD confidence |
| drm_coords | [B,T,5] | Coordenadas no manifold 5D |
| phi_total | [B,T,1] | Saude do manifold |
| vi_direction | [B,T,5] | Vetor de correcao VI |
| temperature | [B,T,1] | Temperatura adaptativa |
| +20 campos opcionais | ... | Eidos, conflito, mood, drives, etc. |

## Instalacao

```bash
# Basico (CPU)
pip install -e ".[dev]"

# Completo (GPU + dados)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,data]"
```

## Testes

```bash
pytest tests/ -v     # 261 testes
```

## Treinamento

### Opcao 1: Script interativo (recomendado)

```powershell
# Windows PowerShell - setup completo interativo
.\train.ps1

# So instala dependencias
.\train.ps1 -Step setup

# So prepara dados
.\train.ps1 -Step data

# So treina (assume tudo pronto)
.\train.ps1 -Step train

# Teste rapido com TinyStories
.\train.ps1 -TestData

# Resume de checkpoint
.\train.ps1 -Step train -Resume checkpoints/350m_rtx4090/step_5000.pt
```

### Opcao 2: Scripts diretos

```bash
# 1. Preparar dados
python scripts/prepare_data.py --dataset fineweb-edu --subset sample-10BT --output data/350m

# 2. Treinar (single GPU)
python scripts/train_distributed.py \
    --config configs/scaling/350m_rtx4090.yaml \
    --data-dir data/350m

# 3. Visualizar
python scripts/plot_training.py --log checkpoints/350m_rtx4090/training_log.json
```

### Opcao 3: Multi-GPU

```bash
# 4 GPUs
torchrun --nproc_per_node=4 scripts/train_distributed.py \
    --config configs/scaling/7b.yaml --data-dir data/7b
```

## Configs de Escala

| Config | Params | VRAM | Hardware |
|--------|--------|------|----------|
| 1m | ~2M | <1 GB | CPU |
| 10m | ~13M | <1 GB | CPU/GPU |
| 50m | ~42M | ~2 GB | 1x GPU |
| 125m | ~110M | ~4 GB | 1x GPU |
| **350m_rtx4090** | **354M** | **~8 GB** | **1x RTX 4090** |
| 1.3b | ~1.3B | ~20 GB | 1x A100 |
| 7b | ~6.6B | ~80 GB | 4x A100 |
| 13b | ~13B | ~160 GB | 8x A100 |
| 30b | ~30B | ~360 GB | 16x A100 |
| 70b | ~70B | ~840 GB | 32x A100 |
| 162b | ~162B | ~2 TB | 64x A100 |
| 250b | ~258B | ~3 TB | 128x H100 |
| 400b | ~391B | ~5 TB | 256x H100 |
| 640b | ~644B | ~8 TB | 512x H100 |

## Continual Learning

EWC (Elastic Weight Consolidation) + Experience Replay integrados no trainer.

```python
from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model
from aletheion_v2.training.trainer import Trainer

config = AletheionV2Config.from_yaml("configs/scaling/350m_rtx4090.yaml")
model = AletheionV2Model(config)
trainer = Trainer(model, config, train_loader, device="cuda")

# Fase 1
trainer.train()
trainer.consolidate_phase()  # Computa Fisher Information Matrix

# Fase 2 (EWC protege conhecimento anterior)
trainer.train_loader = new_loader
trainer.train()
```

## Loss Composta (13 componentes)

```
L = CE + anneal * (VARO + VI + MAD + metric_reg
                 + eidos + conflict + consciousness
                 + grounding + plasticity + frontier
                 + mopsi + contrastive)
```

Annealing: warmup 10% (so CE) -> ramp linear ate 50% -> todas ativas.

## Geracao com Tomografia

```python
from aletheion_v2 import AletheionV2Config, AletheionV2Model
from aletheion_v2.inference.generator import Generator
from aletheion_v2.inference.dashboard_bridge import DashboardBridge

config = AletheionV2Config.from_yaml("configs/scaling/350m_rtx4090.yaml")
model = AletheionV2Model(config)
gen = Generator(model, max_new_tokens=50, top_k=50, use_mpc=True)

result = gen.generate(input_ids)
# result.tomography_per_token: lista de EpistemicTomography
# result.avg_confidence, result.avg_phi

bridge = DashboardBridge()
snapshot = bridge.from_generation_result(result)
print(bridge.to_json(snapshot))
```

## Licenca

Proprietario. Copyright (c) 2025-2026 Felipe Maya Muniz.
