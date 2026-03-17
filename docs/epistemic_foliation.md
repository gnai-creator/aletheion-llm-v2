# 011 - Epistemic Foliation: Preliminary Results

**Data:** 2026-03-17  
**Branch:** `epistemic-foliation` (child of `gravitational_objective`)  
**Modelo:** AletheionV2 1M parâmetros  
**Status:** RESULTADOS PRELIMINARES — validação em escala 350M pendente  

---

## Objetivo

Detectar se o manifold epistêmico 5D do AletheionV2 admite foliation natural, e se a topologia converge para T² (toro) conforme predito pelo DRM (Directional Relational Manifolds).

---

## Configuração Experimental

| Parâmetro | Valor |
|-----------|-------|
| Modelo | AletheionV2 1M parâmetros |
| Hardware | RTX 4090 24GB |
| Tokens extraídos | ~285K por run |
| Subsample homologia | 1500 pontos |
| Seeds K-means | 30 |
| Restarts estabilidade | 10 |

---

## Resultados por Fase de Treino

### Trajetória de Convergência Topológica

| Run | Branch | Métrica Voronoi | H1 | H2 | ANOVA F (médio) | F Score |
|-----|--------|----------------|----|----|-----------------|---------|
| 1 | `full_mahalanobis` | Euclidiana | 48 | 12 | ~260K | 0.3297 |
| 2 | `real_geodesic` (run 1) | G(x) MetricNet | 34 | 5 | ~900K | 0.3108 |
| 3 | `real_geodesic` (run 2) | G(x) MetricNet | 29 | — | ~900K | 0.3108 |

**Observação crítica:** H1 reduz monotonicamente com cada fase de treino e com a ativação de G(x). A direção é consistente com a predição do DRM de convergência toroidal (T² = H1=2, H2=1).

---

## Métricas Detalhadas — Run Final (real_geodesic)

### Foliation Score

```
F = 0.3108
  1 - H(dk)/log(5) = 0.494
  Coherent fraction = 1.000
  Stability ARI     = 0.629
```

### LTSA (Local Tangent Space Analysis)

| Métrica | Modelo real | Null shuffled | Null uniform |
|---------|-------------|---------------|--------------|
| eff_dim média | 3.5 | 4.0 | 4.4 |
| eff_dim mediana | 3 | 4 | 4 |
| distribuição | [0,0,13,10,1] | [0,0,2,19,3] | [0,0,2,14,13] |

**Interpretação:** O manifold real opera em ~3.5D efetivos. Os null models operam em 4.0–4.4D. A compressão dimensional (3.5 vs 4.4) indica estrutura aprendida, não distribuição uniforme no espaço 5D.

### Coerência Tangencial

- **100% dos pares de células** com ângulo principal < 30 graus
- Planos tangentes completamente alinhados entre células vizinhas
- Nota: null models também apresentam 100% — a coerência não discrimina nesta escala de modelo

### Homologia Persistente

| Dimensão | Run 1 (euclidiana) | Run 3 (G(x)) | T² esperado |
|----------|-------------------|--------------|-------------|
| H0 | — | — | 1 |
| H1 | 48 | 29 | **2** |
| H2 | 12 | — | **1** |

**Status:** Topologia complexa (genus > 1). Não atingiu T² nesta escala. Consistente com DRM instável em modelo pequeno com treino limitado.

### Correlação Folha-DRM (ANOVA)

| Eixo DRM | F statistic | p-value |
|----------|-------------|---------|
| q1 (aleatoric) | 835,859 | ~0 (overflow) |
| q2 (epistemic) | 962,552 | ~0 (overflow) |
| q3 (complex) | 808,127 | ~0 (overflow) |
| q4 (familiar) | 781,498 | ~0 (overflow) |
| q5 (confidence) | 1,030,817 | ~0 (overflow) |

**Interpretação:** Cada célula Voronoi corresponde a uma região DRM completamente distinta em todos os 5 eixos. F > 800K indica separação epistêmica extrema — as folhas têm perfis interpretativos próprios.

### Estabilidade

- ARI médio: **0.629** (std=0.064, 45 pares)
- Abaixo do threshold publicável (0.7) — esperado para modelo 1M
- Indica estrutura parcialmente estável, não arbitrária

---

## Reeb Graphs

| Função | std raw | std pós-logit | Resultado |
|--------|---------|---------------|-----------|
| confidence | 0.0147 | 0.1357 | 64 nodes, 113 edges — variação insuficiente |
| phi_total | 0.0034 | 0.0144 | 227 nodes, 1776 edges — essencialmente constante |

**Limitação confirmada:** phi_total e confidence são funções saturadas em modelo 1M. Não servem como funções de Morse para foliation via level sets nesta escala. Esperado resolver em 350M com mais steps de treino.

---

## Interpretação dos Resultados

### O que está confirmado

1. **Estrutura real, não ruído:** eff_dim 3.5 vs null 4.4 — o manifold é comprimido abaixo da dimensionalidade do espaço ambiente
2. **G(x) melhora a geometria:** Ativação da MetricNet reduz H1 (48→29) e aumenta F ANOVA (~260K→~900K) simultaneamente
3. **Folhas epistêmicas distintas:** ANOVA F>800K em todos os eixos confirma que cada região Voronoi tem identidade epistêmica própria
4. **Direção toroidal:** Simplificação monotônica de H1 com progressão do treino é consistente com a predição do DRM

### O que não está confirmado

1. **T² não atingido:** H1=29 vs H1=2 esperado — convergência em andamento, não completa
2. **Estabilidade abaixo do threshold:** ARI=0.629 < 0.7
3. **Reeb graph limpo:** phi_total saturado impede análise de level sets

### Hipótese de trabalho

A topologia complexa observada (H1=29) é consistente com DRM em estado de convergência pré-toroidal. O Theorem 8.1 do DRM prediz convergência para T² em DRMs **estáveis**. O modelo 1M com treino limitado e sem `gravitational_objective` ativo não atingiu estabilidade. A validação definitiva requer a cadeia completa em 350M.

---

## Próximos Passos

### Sequência de treino 350M (H200)

```
backbone (350M, congelado)
    → full_mahalanobis   (mês 1, fase 1)
    → real_geodesic      (mês 1, fase 2)
    → gravitational_obj  (mês 2, fase 3)
    → epistemic-foliation (mês 2, fase 4 — inferência)
```

### Critérios de validação definitiva

| Critério | Threshold | Status atual |
|----------|-----------|--------------|
| H1 long bars | = 2 | 29 |
| H2 long bars | = 1 | — |
| ARI estabilidade | > 0.7 | 0.629 |
| F score | > 0.5 | 0.310 |
| phi_total std | > 0.5 | 0.003 |

### Hipótese falsificável

Se após `gravitational_objective` em 350M com G(x) ativa:
- H1 ≥ 10: DRM não converge para T² na arquitetura atual
- H1 = 2 com barras longas: DRM validado empiricamente

---

## Arquivos Gerados

```
eval_results/foliation_1m/
├── full_mahalanobis/
│   ├── full_mahalanobis_vectors.npy      # 284928 x 5
│   ├── full_mahalanobis_drm_coords.npy
│   ├── voronoi_labels.npy
│   ├── voronoi_centers.npy
│   ├── eigenvalues.npy
│   ├── eff_dims.npy
│   ├── coherence_matrix.npy
│   ├── reeb_confidence.json
│   ├── reeb_phi.json
│   ├── homology.json                     # H1=48, H2=12
│   └── foliation_results.json            # F=0.3297
└── real_geodesic/
    ├── real_geodesic_vectors.npy         # 285696 x 5
    ├── real_geodesic_drm_coords.npy
    ├── homology.json                     # H1=29
    └── foliation_results.json            # F=0.3108
```

---

## Referências

- DRM: Directional Relational Manifolds — DOI: 10.5281/zenodo.19058837
- AletheionV2 — DOI: 10.13140/RG.2.2.11471.14241
- ATIC framework — DOI: 10.5281/zenodo.19058926
- The Geometry of Consciousness — DOI: 10.5281/zenodo.19059445

---

*Documento gerado em 2026-03-17. Resultados preliminares — não citar como validação definitiva.*