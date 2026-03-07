# 005 - Continual Learning + Configs de Escala

**Data:** 2026-03-07
**Status:** Concluido
**Testes:** 261/261 passing (51 novos)

## Resumo

Implementacao de Continual Learning (EWC + Experience Replay) no pipeline
de treinamento, config otimizado para RTX 4090, verificacao de configs
existentes e criacao de configs ate 640B.

Baseado no survey: Wang et al. "A Comprehensive Survey of Continual
Learning: Theory, Method and Application" (2302.00487v3).

## Modulos Criados

### EWC - Elastic Weight Consolidation

**Arquivo:** `src/aletheion_v2/training/ewc.py` (~180 linhas)

- `EWCRegularizer(nn.Module)`: Regularizador com Fisher Information Matrix diagonal
- Computa Fisher via gradientes da log-verossimilhanca empirica
- Online EWC: acumula Fisher de multiplas fases com decay exponencial
- L_ewc = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
- State dict para checkpoint: Fisher + parametros de referencia
- Estatisticas de importancia para monitoramento

### Experience Replay Buffer

**Arquivo:** `src/aletheion_v2/training/replay_buffer.py` (~170 linhas)

- `ReplayBuffer`: Reservoir sampling com memoria fixa
- Armazena pares (input_ids, labels) em CPU
- `mix_batch()`: Substitui fracao do batch atual com amostras do buffer
- Lida com seq_len diferentes (trunca/pad automatico)
- State dict para checkpoint

## Config Adicionados

| Campo | Tipo | Default | Descricao |
|-------|------|---------|-----------|
| enable_ewc | bool | False | Habilita EWC |
| ewc_lambda | float | 100.0 | Peso da regularizacao |
| ewc_online | bool | True | Acumula Fisher |
| ewc_gamma | float | 0.9 | Decay de fases anteriores |
| ewc_fisher_samples | int | 256 | Amostras para Fisher |
| enable_replay | bool | False | Habilita replay |
| replay_buffer_size | int | 10000 | Tamanho do buffer |
| replay_mix_ratio | float | 0.1 | Fracao de replay no batch |

## Integracao nos Trainers

### Trainer (single-GPU)
- EWC + replay instanciados no __init__
- `train_epoch()`: replay.add + mix_batch + EWC loss no backward
- `consolidate_phase()`: computa Fisher apos cada fase
- Checkpoint salva/carrega EWC + replay state

### DistributedTrainer (multi-GPU)
- Mesma integracao com amp_context no Fisher
- EWC loss dividida por gradient_accumulation_steps
- `consolidate_phase()` com log condicional (is_main)

## Config RTX 4090

**Arquivo:** `configs/scaling/350m_rtx4090.yaml`

| Parametro | Valor |
|-----------|-------|
| d_model | 1024 |
| n_heads | 16 (head_dim=64) |
| n_layers | 24 |
| d_ff | 4096 |
| max_seq_len | 1024 |
| batch_size | 16 |
| mixed_precision | bf16 |
| gradient_checkpointing | true |
| gradient_accumulation | 4 |
| compile_model | true |
| enable_ewc | true |
| enable_replay | true |

**Estimativa de memoria:**
- Modelo bf16: ~670MB
- Optimizer fp32: ~4.0GB
- Activations (checkpointed): ~2.0GB
- Total: ~8.9GB (24GB - 15.1GB livre)

**Throughput estimado:** ~12K-18K tokens/s com torch.compile

## Verificacao de Configs Existentes

Todas as 11 configs (1m-162b) verificadas:

| Config | d_model | n_heads | head_dim | d_ff | Status |
|--------|---------|---------|----------|------|--------|
| 1m | 64 | 2 | 32 | 256 | OK |
| 10m | 256 | 4 | 64 | 1024 | OK |
| 50m | 512 | 8 | 64 | 2048 | OK |
| 125m | 768 | 12 | 64 | 3072 | OK |
| 350m | 1024 | 16 | 64 | 4096 | OK |
| 1.3b | 2048 | 16 | 128 | 8192 | OK |
| 7b | 4096 | 32 | 128 | 11008 | OK |
| 13b | 5120 | 40 | 128 | 13824 | OK |
| 30b | 6656 | 52 | 128 | 17920 | OK |
| 70b | 8192 | 64 | 128 | 28672 | OK |
| 162b | 12288 | 96 | 128 | 49152 | OK |

## Configs Novas (3)

| Config | d_model | n_heads | n_layers | d_ff | Params Est. |
|--------|---------|---------|----------|------|-------------|
| 250b | 16384 | 128 | 80 | 65536 | ~258B |
| 400b | 18432 | 144 | 96 | 73728 | ~391B |
| 640b | 20480 | 160 | 128 | 81920 | ~644B |

## Conceitos de CL Aplicados

Do paper Wang et al. (2302.00487v3):

1. **EWC (Regularization-based)**: Fisher diagonal penaliza mudancas em
   parametros importantes - Eq. 12 do paper
2. **Experience Replay (Replay-based)**: Reservoir sampling mantem amostra
   uniforme de dados anteriores para rehearsal
3. **Online EWC**: Acumula Fisher com decay exponencial para multiplas fases
4. **Estabilidade-Plasticidade**: Lambda EWC controla trade-off entre
   reter conhecimento (estabilidade) e aprender novo (plasticidade)

## Arquivos Criados (5)

```
src/aletheion_v2/training/ewc.py
src/aletheion_v2/training/replay_buffer.py
configs/scaling/350m_rtx4090.yaml
configs/scaling/250b.yaml
configs/scaling/400b.yaml
configs/scaling/640b.yaml
tests/test_continual_learning.py
```

## Arquivos Modificados (3)

| Arquivo | Mudancas |
|---------|----------|
| config.py | +8 campos CL |
| training/trainer.py | +EWC/replay init, mix_batch, EWC loss, consolidate_phase |
| training/trainer_distributed.py | +EWC/replay, consolidate_phase, checkpoint |

## Uso

```python
# Treinamento continuo com EWC + Replay
config = AletheionV2Config.from_yaml("configs/scaling/350m_rtx4090.yaml")
model = AletheionV2Model(config)
trainer = Trainer(model, config, train_loader, eval_loader, device="cuda")

# Fase 1: dados iniciais
trainer.train()
trainer.consolidate_phase()  # Computa Fisher

# Fase 2: novos dados (EWC protege conhecimento anterior)
trainer.train_loader = new_loader
trainer.train()
trainer.consolidate_phase()  # Acumula Fisher (online)
```
