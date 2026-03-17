# 010 - Riemannian Voronoi Tessellation and Foliation Detection

**Data:** 2026-03-17
**Branch:** epistemic-foliation (child of gravitational_objective)
**Status:** PLAN (not implemented)

## Objetivo

Detectar se o manifold epistemico 5D admite uma foliacao natural — i.e., se as celulas Voronoi se decompoe em folhas de dimensao menor correspondendo a estados epistemicos interpretaveis.

## 1. Extracao de Vetores Epistemicos

Coletar 5 escalares por token do output do modelo:

| Index | Campo | Fonte em EpistemicTomography | Range |
|-------|-------|------------------------------|-------|
| 0 | q1 (aleatoric) | tomography.q1 | [0,1] sigmoid |
| 1 | q2 (epistemic) | tomography.q2 | [0,1] sigmoid |
| 2 | confidence | tomography.confidence | [0,1] Gaussian decay |
| 3 | vi_magnitude | tomography.vi_direction.norm(dim=-1) | [0, ~0.6] |
| 4 | cognitive_state | tomography.phi_total | [0,1] weighted sum |

**Script:** `scripts/extract_epistemic_vectors.py` seguindo padrao de `eval_epistemic.py` (load_model, prepare_eval_tokens, batched inference com return_tomography=True).

**Dataset:** ~500K-2M tokens. Armazenar tambem token IDs para analise de interpretabilidade.

**Nota:** O epistemic head forca fp32 internamente (model.py:163-171) mesmo com autocast bf16.

## 2. Metrica Riemanniana para Voronoi

### Opcao A: MetricNet G(x) aprendida (RECOMENDADA)

O modelo ja aprende um tensor metrico SPD posicao-dependente via MetricNet (metric_tensor.py:130-335). Vantagens:

- G(x) codifica a geometria que o proprio modelo aprendeu
- SPD por construcao (Cholesky com softplus diagonal)
- Varia com posicao — geometria Riemanniana genuina
- `line_integral_distance()` ja implementa quadratura Gauss-Legendre

**Custo:** N=500K pontos x K=50 seeds x 5 GL points = 125M avaliacoes do MetricNet (MLP 5->32->15). Factivel em GPU batched (~minutos).

### Opcao B: Fisher-Rao (baseline)

Metrica constante estimada da covariancia empirica. Perde informacao de posicao.

### Opcao C: Mahalanobis constante (baseline)

LearnableMetricTensor (metric_tensor.py:39-128). Plano, sem curvatura.

**Decisao:** Usar A como primaria, C como baseline. Se foliacao aparece sob G(x) mas nao sob G constante, evidencia que curvatura aprendida codifica geometria epistemica.

## 3. Deteccao de Foliacao

### 3.1 Selecao de Seeds

Seeds iniciais: 6 anchors de AnchorPoints (manifold_embedding.py:38-61):
- truth: [0.1, 0.1, 0.5, 0.9, 0.9]
- ignorance: [0.3, 0.9, 0.5, 0.5, 0.2]
- noise: [0.9, 0.3, 0.5, 0.5, 0.3]
- complex: [0.5, 0.5, 0.9, 0.5, 0.5]
- stale: [0.3, 0.3, 0.5, 0.1, 0.4]
- ideal: [0.2, 0.2, 0.3, 0.8, 0.8]

Aumentar com K-means Riemanniano (K=20-50) sob distancia G(x). K-means++ com distancia Riemanniana para inicializacao.

### 3.2 Local Tangent Space Analysis (LTSA)

Para cada celula Voronoi V_k:
1. Coletar N_k pontos
2. PCA local (ponderado por G(x) no centroide)
3. Espectro de eigenvalues {lambda_1 >= ... >= lambda_5}
4. Gap no espectro: se lambda_d >> lambda_{d+1}, celula e d-dimensional

**Criterio de foliacao:** Se >80% das celulas tem dimensao efetiva d_k = d para algum d fixo, sugere foliacao de codimensao (5-d).

### 3.3 Teste de Coerencia Tangencial

Para cada par de celulas vizinhas (V_i, V_j):
1. Subespacos principais S_i, S_j (top-d autovetores)
2. Angulo principal via SVD da cross-product matrix
3. Angulo maximo < 30 graus = coerencia tangencial

Foliacao suportada se grafo de coerencia tem componente conexa grande.

### 3.4 Analise Reeb Graph / Level Sets

Candidatos para funcao de foliacao f: M -> R:
- f = confidence
- f = phi_total

Para cada candidato, computar grafo Reeb:
1. Binning de f em 50-100 niveis
2. Componentes conexas por nivel (usando grafo de vizinhanca Riemanniana)
3. Tracking de merges/splits entre niveis

Se Reeb graph e arvore (sem loops), level sets definem foliacao simples.

### 3.5 Persistent Homology (sanity check)

H_0, H_1 do point cloud com distancia Riemanniana:
- H_0: estrutura de clusters = folhas em dado nivel
- H_1: loops correspondem a topologia toroidal esperada (Theorem 8.1)

Usar ripser ou giotto-tda em subsample (~10K-50K pontos).

## 4. Visualizacao

### 4.1 Projecao T^2 Toroidal

Overlay de celulas Voronoi como cor no toro (q1=u, q2=v). Mostra se estrutura Voronoi e visivel na projecao que o TopologicalWatchdog usa.

### 4.2 UMAP com Pre-metrica Riemanniana

2D embedding com distancia customizada G(x). Colorir por:
- (a) celula Voronoi
- (b) dimensionalidade efetiva d_k
- (c) confidence (colormap continuo)
- (d) tipo de token (funcao vs conteudo vs pontuacao)

### 4.3 Espectro de Eigenvalues

Para cada celula, plotar 5 eigenvalues ordenados. Overlay de todas as celulas. Foliacao = gap consistente no mesmo indice.

### 4.4 Heatmap de Coerencia Tangencial

Heatmap celula-por-celula do angulo principal. Estrutura bloco-diagonal indica foliacao.

### 4.5 Reeb Graph

Nos dimensionados por numero de pontos, arestas coloridas por valor do level set.

**Tema visual:** dark theme (facecolor="#0d1117") seguindo padrao de eval_epistemic.py.

## 5. Criterios de Validacao

### 5.1 Modelos Nulos

Tres null distributions:
1. **Shuffled:** permutar cada dimensao independentemente (destroi correlacoes, preserva marginais)
2. **Uniform:** N pontos uniformes em [0,1]^5
3. **Backbone:** mesmos vetores do checkpoint backbone (antes do fine-tuning epistemico)

Para cada null: rodar pipeline identico. P-values via:
- KS test para concentracao de d_k
- Permutation test para coerencia media
- Contagem de pontos criticos no Reeb graph

Se foliacao aparece so apos fine-tuning, o treino epistemico a criou.

### 5.2 Estabilidade sob Perturbacao de Seeds

10 restarts aleatorios de K-means. ARI > 0.7 = estrutura estavel. ARI < 0.3 = arbitraria.

### 5.3 Teste de Interpretabilidade

Para cada celula/folha:
- Media e std de cada dimensao epistemica
- Top-10 tokens mais frequentes
- Perplexidade media

Foliacao nao-arbitraria deve produzir perfis interpretaveis (ex: "folha factual" = low q1, low q2, high confidence, high phi).

### 5.4 Teste de Dependencia da Metrica

Comparar sob G(x) vs G constante vs identidade. Se robusta sob todas = reflete distribuicao dos dados. Se so sob G(x) = reflete geometria aprendida.

### 5.5 Score de Foliacao

```
F = (1 - H(d_k)/log(5)) * mean_coherence * stability_ARI
```

- H(d_k): entropia da distribuicao de dimensionalidades (baixa = concentrada = bom)
- mean_coherence: alinhamento medio dos planos tangentes
- stability_ARI: estabilidade sob re-seeding

F in [0, 1]. Reportar F com p-value do null comparison.

## Sequenciamento

| Fase | Script | Tempo GPU |
|------|--------|-----------|
| 1 | extract_epistemic_vectors.py | ~30 min |
| 2 | voronoi_foliation.py (seeds + tessellation) | ~15 min |
| 3 | voronoi_foliation.py (LTSA + coherence + Reeb) | ~10 min CPU |
| 4 | voronoi_foliation.py (null models) | ~45 min |
| 5 | plot_foliation.py | ~5 min |

## Dependencias Externas

numpy, scipy, scikit-learn, umap-learn, ripser/giotto-tda (opcional), matplotlib

## Desafios

1. **Custo MetricNet:** batching resolve (MLP pequeno)
2. **Celulas pequenas:** minimo 100 pontos por celula para PCA estavel
3. **Saturacao sigmoid:** checar marginais; considerar logit transform (mas muda geometria)
4. **Topologia toroidal vs flat:** Voronoi opera em 5D, nao no toro. Se foliacao alinha com T^2, valida ambos.

## Arquivos Criticos

- `src/aletheion_v2/core/model.py` — forward com return_tomography=True
- `src/aletheion_v2/drm/metric_tensor.py` — MetricNet.forward(), line_integral_distance()
- `src/aletheion_v2/drm/manifold_embedding.py` — AnchorPoints, ManifoldEmbedding
- `scripts/eval_epistemic.py` — padrao para load_model, prepare_eval_tokens
- `src/aletheion_v2/core/output.py` — EpistemicTomography dataclass
