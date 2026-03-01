# 003 - Graficos de Treinamento e Download de Datasets

**Data:** 2026-02-28
**Autor:** Claude (assistido)

## Resumo

Adicionado sistema completo de visualizacao de treinamento e download de datasets separados por escala.

## Arquivos Adicionados/Modificados

### Scripts
- `scripts/plot_training.py` (~470 LOC) - Gera 9 tipos de graficos:
  1. **loss_curves.png** - Train + Eval loss com suavizacao
  2. **learning_rate.png** - Schedule do LR (warmup + cosine)
  3. **epistemic_metrics.png** - Q1, Q2, Confidence, Phi (2x2 grid)
  4. **loss_components.png** - CE, VARO, VI, MAD em escala log
  5. **throughput.png** - Tokens/s e Gradient Norm
  6. **perplexity.png** - Perplexidade (exp(loss))
  7. **drm_manifold.png** - Geodesic distance, dim_D, VI severity
  8. **dashboard.png** - Todos os graficos em 3x4 grid unico
  9. **comparison.png** - Comparacao entre escalas (loss, PPL, confidence)
- `scripts/train_and_plot.py` (~80 LOC) - Treina e gera graficos automaticamente

### Trainer Modificado
- `trainer_distributed.py` - Adicionado:
  - Extracao de metricas epistemicas (q1, q2, confidence, phi, dim_d, vi_severity, geodesic_distance)
  - Tracking de grad_norm
  - Salvamento automatico de `training_log.json` ao final do treinamento
  - Formato JSON compativel com plot_training.py

### Datasets Preparados
- `data/1m/` - TinyStories, 10M tokens, 1 shard
- `data/10m/` - TinyStories, 200M tokens (download em andamento)
- `data/50m/` - OpenWebText, 1B tokens (download em andamento)
- `data/125m/` - OpenWebText, 2.5B tokens (pendente)

### Graficos Demo
- `plots/demo/` - 8 graficos com dados sinteticos (125M simulado)
- `plots/comparison/` - Comparacao 1M/10M/50M/125M + dashboards individuais

## Uso

```bash
# Graficos de demonstracao
python scripts/plot_training.py --demo --output plots/demo

# Comparacao entre escalas
python scripts/plot_training.py --demo-compare --output plots/comparison

# Apos treinamento real
python scripts/plot_training.py --log checkpoints/training_log.json --output plots/

# Treinar e gerar graficos automaticamente
python scripts/train_and_plot.py --config configs/scaling/1m.yaml --data-dir data/1m
```

## Testes
85/85 testes passando (nenhuma regressao).
