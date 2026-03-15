# 010 - Experimento de Curvatura: 3 Branches

Data: 2026-03-15

## Objetivo

Comparar tres hipoteses sobre a geometria do manifold epistemico:

| Branch | Metrica | Curvatura | Hipotese |
|--------|---------|-----------|----------|
| `main` | Diagonal (BayesianTau) | Zero | Eixos independentes sao suficientes |
| `full_mahalanobis` | G constante 5x5 (Cholesky) | Zero (plano obliquo) | Correlacoes entre eixos melhoram calibracao |
| `real_geodesic` | G(x) variavel (MetricNet) | Real (Riemanniana) | Geometria local distinta por regiao |

## Sequencia de Treino

### 1. full_mahalanobis (primeiro)

```bash
git checkout full_mahalanobis

torchrun --nproc_per_node=5 scripts/train_distributed.py \
  --config configs/scaling/350m_full_mahalanobis_5xh200.yaml \
  --resume checkpoints/350m_backbone_final/final.pt
```

- Menor risco: mesma arquitetura do main com G completo
- Se G convergir para diagonal, experimento termina aqui
- Tempo estimado: ~5-6h (5xH200), custo ~$108

### 2. real_geodesic (so se full_mahalanobis mostrar estrutura off-diagonal)

```bash
git checkout real_geodesic

torchrun --nproc_per_node=5 scripts/train_distributed.py \
  --config configs/scaling/350m_real_geodesic_5xh200.yaml \
  --resume checkpoints/350m_backbone_final/final.pt
```

- MetricNet: MLP 5->32->15, ~700 params, LR 10x base
- Integral de linha Gauss-Legendre (5 pontos)
- ~10-15% mais lento que full_mahalanobis
- Tempo estimado: ~6-7h (5xH200), custo ~$126

## Teste de Curvatura

Apos cada treino, rodar o teste na respectiva branch:

```bash
# Branch main (baseline, pode rodar agora com checkpoint v2):
git checkout main
PYTHONPATH=src python scripts/test_curvature.py \
  --checkpoint checkpoints/350m_epistemic_finetune_v2/final.pt \
  --branch main

# Branch full_mahalanobis (apos treino):
git checkout full_mahalanobis
PYTHONPATH=src python scripts/test_curvature.py \
  --checkpoint checkpoints/350m_full_mahalanobis/final.pt \
  --branch full_mahalanobis

# Branch real_geodesic (apos treino):
git checkout real_geodesic
PYTHONPATH=src python scripts/test_curvature.py \
  --checkpoint checkpoints/350m_real_geodesic/final.pt \
  --branch real_geodesic

# Comparar os 3:
PYTHONPATH=src python scripts/test_curvature.py --compare
```

Resultados salvos em `eval_results/curvature/{branch}.json`.

## O que Medir

### Probes

- **high_confidence**: "The capital of France is", "2 + 2 =", etc.
- **low_confidence**: "The GDP of Nauru in 2019 was", etc.
- **context_sensitive**: pares ambiguos (bank/plant/bat) com semantica diferente

### Metricas por Branch

- **main**: distancia diagonal d = sqrt(sum((xi-yi)^2 / tau_i^2))
- **full_mahalanobis**: distancia Mahalanobis d = sqrt(delta^T G delta)
- **real_geodesic**: integral de linha d = int sqrt(dx^T G(gamma(t)) dx) dt

### Variacao de G (so real_geodesic)

- G(x) avaliado em 20 pontos ao longo do path high->low confidence
- Se variacao maxima > 0.01: espaco CURVO
- Se variacao maxima ~ 0: espaco PLANO

## Criterio de Decisao

| Resultado | Interpretacao |
|-----------|---------------|
| geodesic ~ diagonal em tudo | Espaco plano, diagonal correta |
| geodesic > diagonal nos pares ambiguos, ~ nos domesticos | Curvatura local real em fronteiras semanticas |
| geodesic > diagonal uniformemente | G(x) aprendeu escala mas nao estrutura |
| variacao de G alta no path | Curvatura Riemanniana confirmada |

## Configs

- `configs/scaling/350m_full_mahalanobis_5xh200.yaml`
- `configs/scaling/350m_real_geodesic_5xh200.yaml`

Ambas baseadas na v2 confirmada: 1B tokens, fp32, lambda_decay_k=0.0001,
loss_warmup=0.30, loss_ramp=0.80, grad_clip=0.5.

## Backbone

`checkpoints/350m_backbone_final/final.pt` (step 106811, 7B tokens)
