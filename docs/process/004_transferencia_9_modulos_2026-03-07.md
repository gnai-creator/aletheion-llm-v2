# 004 - Transferencia de 9 Modulos ATIC -> Aletheion LLM v2

**Data:** 2026-03-07
**Status:** Concluido
**Testes:** 210/210 passing

## Resumo

Transferencia de 9 modulos de orquestracao do ATIC para o aletheion-llm-v2,
todos convertidos de logica deterministica para nn.Module treinaveis com
backward-compatible Optional fields.

## Modulos Implementados

### Tier 1 - Alto Impacto (~1.7K params)

| Modulo | Arquivo | Params | Descricao |
|--------|---------|--------|-----------|
| EidosDecay | `eidos/eidos_decay.py` | ~368 | Balanceamento de eixos 5D (decay/reinforce) |
| Filosofia3 | `filosofia3/conflict_head.py` | ~373 | Deteccao conflito phi-psi (blend analitico/aprendivel) |
| SelfModel | `consciousness/self_model_head.py` | ~982 | Humor, curiosidade, energia, 3 drives |

### Tier 2 - Alto Impacto, Medio Esforco (~112K params)

| Modulo | Arquivo | Params | Descricao |
|--------|---------|--------|-----------|
| TaskHead | `grounding/task_head.py` | ~49.9K | Classificacao 9 tipos de tarefa |
| AmbiguityHead | `grounding/ambiguity_head.py` | ~49.5K | Deteccao 5 tipos de ambiguidade |
| PlasticityGate | `plasticity/plasticity_gate.py` | ~12.3K | Gate de plasticidade com deplecao cumulativa |
| FrontierHead | `mpl/frontier_head.py` | ~129 | Scoring de fronteira para exploracao |
| DensityTracker | `mpl/density_tracker.py` | 0 (utility) | Hash-grid esparso 5D |

### Tier 3 - Medio Impacto (~250K params)

| Modulo | Arquivo | Params | Descricao |
|--------|---------|--------|-----------|
| HumanStateHead | `mopsi/human_state_head.py` | ~25.1K | Estado humano 5D + psi |
| PhiPsiMediator | `mopsi/human_state_head.py` | ~41 | Mediacao phi-psi |
| StateConditioning | `causal_state/state_conditioning.py` | ~25.5K | Condicionamento causal (state_vector) |
| PolicyBinding | `causal_state/state_conditioning.py` | 0 (heuristico) | Mapeamento estado -> params geracao |
| ContrastiveHead | `metacognitive/contrastive_head.py` | ~200K | Auto-avaliacao contrastiva dual |

## Arquivos Criados (20)

```
src/aletheion_v2/eidos/__init__.py
src/aletheion_v2/eidos/eidos_decay.py
src/aletheion_v2/filosofia3/__init__.py
src/aletheion_v2/filosofia3/conflict_head.py
src/aletheion_v2/consciousness/__init__.py
src/aletheion_v2/consciousness/self_model_head.py
src/aletheion_v2/grounding/__init__.py
src/aletheion_v2/grounding/task_head.py
src/aletheion_v2/grounding/ambiguity_head.py
src/aletheion_v2/plasticity/__init__.py
src/aletheion_v2/plasticity/plasticity_gate.py
src/aletheion_v2/mpl/__init__.py
src/aletheion_v2/mpl/density_tracker.py
src/aletheion_v2/mpl/frontier_head.py
src/aletheion_v2/mopsi/__init__.py
src/aletheion_v2/mopsi/human_state_head.py
src/aletheion_v2/causal_state/__init__.py
src/aletheion_v2/causal_state/state_conditioning.py
src/aletheion_v2/metacognitive/__init__.py
src/aletheion_v2/metacognitive/contrastive_head.py
```

### Loss files (8)

```
src/aletheion_v2/loss/eidos_loss.py
src/aletheion_v2/loss/conflict_loss.py
src/aletheion_v2/loss/consciousness_loss.py
src/aletheion_v2/loss/grounding_loss.py
src/aletheion_v2/loss/plasticity_loss.py
src/aletheion_v2/loss/frontier_loss.py
src/aletheion_v2/loss/mopsi_loss.py
src/aletheion_v2/loss/contrastive_loss.py
```

### Testes (6)

```
tests/test_eidos.py
tests/test_filosofia3.py
tests/test_consciousness.py
tests/test_grounding.py
tests/test_plasticity.py
tests/test_mpl.py
tests/test_mopsi.py
tests/test_causal_state.py
tests/test_metacognitive.py
```

## Arquivos Modificados (5)

| Arquivo | Mudancas |
|---------|----------|
| `config.py` | +35 campos config + 9 enable flags + 8 lambda weights |
| `core/output.py` | +20 campos Optional no EpistemicTomography, to_dict/detach/to genericos |
| `epistemic/epistemic_head.py` | +3 _init_tier*() + forward expandido (~280 linhas) |
| `loss/composite_loss.py` | +_init_extension_losses() + _compute_extension_losses() + 8 lambdas |
| `core/model.py` | +state_vector e dream_mode no forward() |

## Config Flags

Todos habilitados por default (`True`):
- `enable_eidos`, `enable_filosofia3`, `enable_consciousness`
- `enable_grounding`, `enable_plasticity`, `enable_mpl`
- `enable_mopsi`, `enable_causal_state`, `enable_metacognitive`

## Loss Weights

| Lambda | Valor | Loss |
|--------|-------|------|
| lambda_eidos | 0.005 | Balanceamento de eixos |
| lambda_conflict | 0.005 | Conflito phi-psi |
| lambda_consciousness | 0.003 | Energia minima |
| lambda_grounding | 0.005 | Entropia + calibracao |
| lambda_plasticity | 0.002 | Plasticidade minima |
| lambda_frontier | 0.002 | Exploracao fronteira |
| lambda_mopsi | 0.003 | Alinhamento psi-confianca |
| lambda_contrastive | 0.003 | Anti-colapso contrastivo |

## Overhead

- Params adicionais: ~364K (~0.30% do modelo medium 120M)
- Backward-compatible: todos os campos novos sao Optional
- Annealing: mesmo schedule das losses core (warmup -> ramp -> full)
