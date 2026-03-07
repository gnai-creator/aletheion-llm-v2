# Arte Anterior e Trabalhos Relacionados

Este documento lista a origem dos conceitos implementados no Aletheion LLM v2
e os trabalhos academicos que fundamentam a teoria subjacente.

---

## Conceitos Originais (Felipe Maya Muniz)

Os seguintes conceitos foram **criados, projetados e implementados originalmente
por Felipe Maya Muniz** no projeto ATIC (Adaptive Turing Intelligent Cognition,
2025-2026). No ATIC, operam como modulos Python num pipeline de orquestracao.
No AletheionV2, foram transpostos para `nn.Module` treinaveis dentro da rede neural.

| Conceito | Descricao | Origem |
|----------|-----------|--------|
| **DRM (Directional Relational Manifold)** | Manifold Riemanniano 5D para modelagem epistemica com tensor metrico SPD, geodesicas, curvatura de Ricci e campo direcional | ATIC v2.0+ (2025) |
| **MAD (Metric-Aware Distance)** | Confianca Bayesiana com tau^2 aprendivel, prior InverseGamma, decaimento Gaussiano anisotropico e mixture models | ATIC v2.0+ (2025) |
| **VI (Vetor de Intencionalidade)** | Monitoramento e correcao homestatica do manifold via phi(M) com 4 componentes (dimensional, dispersao, entropia, confianca), hysteresis e cooldown | ATIC v2.5+ (2025) |
| **MPC Navigator** | Controle preditivo com beam search (K=4, D=3) no manifold epistemico, 12 acoes de intervencao, planejamento de trajetorias otimas | ATIC v3.0+ (2025) |
| **MOPsi (Modulo Orientador de Psi)** | Estado humano 5D + mediacao phi-psi + SynergyFinder | ATIC v3.0+ (2025) |
| **MPL (Modulo Projetor de Longo Prazo)** | Density tracking + frontier exploration + scoring de novidade | ATIC v3.0+ (2025) |
| **EidosDecay** | Decay invertido + dream cycles para balanceamento dos 5 eixos do manifold | ATIC v2.5+ (2025) |
| **Filosofia3** | Conflito phi-psi negociado com deteccao por cosine similarity | ATIC v3.0+ (2025) |
| **SelfModel** | Modelo de consciencia com humor, curiosidade, energia e drives motivacionais | ATIC v2.0+ (2025) |
| **Termodinamica Computacional** | Custo de Landauer, plasticidade sinaptica com deplecao, dinamicas irreversiveis | ATIC v3.0+ (2025) |
| **Tomografia Epistemica** | Framework de analise multi-dimensional por token com 30+ campos epistemicos | ATIC v2.0+ (2025) |
| **Tri-Brain Consensus** | Arquitetura de consenso Alpha/Beta/Gamma com threshold configuravel | ATIC v1.0+ (2025) |
| **Pipeline Epistemico (50 steps)** | Pre/post-processing com routing epistemico, grounding, verificacao, governanca | ATIC v2.0+ (2025) |
| **Causal Complexity Framework** | Transfer Entropy, KL Divergence, Granger Causality, Mutual Information, phi-operacional (proxy IIT) | ATIC v3.5+ (2026) |
| **GNN Routing (REINFORCE+Adam)** | Grafo neural para selecao de agentes com policy optimization (~3,475 params) | ATIC v3.0+ (2025) |
| **Dynamic Associative Memory (DAM)** | Memoria hibrida BM25 + vetorial + RRF com reindex e poda | ATIC v3.5+ (2026) |

**Repositorio ATIC:** [github.com/gnai-creator/atic_consulting](https://github.com/gnai-creator/atic_consulting)

**Nota:** A transposciao de orquestracao-em-Python para modulos-neurais-treinaveis
(nn.Module com gradientes end-to-end) e a contribuicao central do AletheionV2.

---

## Trabalhos Academicos Relacionados

Os trabalhos abaixo fornecem a fundamentacao teorica e matematica sobre a qual
os conceitos originais se apoiam.

### Arquitetura Transformer

- **Vaswani et al. (2017)** - "Attention Is All You Need"
  Arquitetura Transformer original com multi-head self-attention.

- **Radford et al. (2019)** - "Language Models are Unsupervised Multitask Learners" (GPT-2)
  Decoder-only com pre-norm residual. Base arquitetural do Aletheion.

- **Touvron et al. (2023)** - "LLaMA: Open and Efficient Foundation Language Models"
  RoPE, SwiGLU feed-forward, pre-norm. Padroes adotados no Aletheion.

### Incerteza Epistemica

- **Der Kiureghian & Ditlevsen (2009)** - "Aleatory or epistemic? Does it matter?"
  Distincao formal entre incerteza aleatoria (Q1) e epistemica (Q2).
  Fundamenta a decomposicao Q1/Q2 no sistema epistemico original do ATIC.

- **Gal & Ghahramani (2016)** - "Dropout as a Bayesian Approximation"
  MC Dropout para estimacao de incerteza. O ATIC optou por gates treinaveis
  (Q1Gate/Q2Gate) em vez de dropout, o que e uma abordagem original.

- **Lakshminarayanan et al. (2017)** - "Simple and Scalable Predictive Uncertainty Estimation"
  Deep ensembles. Motivacao para a abordagem alternativa via manifold (DRM)
  criada no ATIC.

### Geometria Riemanniana em ML

- **Bronstein et al. (2017)** - "Geometric Deep Learning"
  Framework para deep learning em dominios nao-Euclidianos.
  O DRM do ATIC estende estes conceitos com o campo direcional D(p)
  e o manifold epistemico 5D, que sao contribuicoes originais.

- **Nickel & Kiela (2017)** - "Poincare Embeddings for Learning Hierarchical Representations"
  Embeddings em espacos hiperbolicos. O DRM usa geometria Riemanniana geral
  (nao restrita a Poincare) com tensor metrico aprendivel, que e original.

### Model Predictive Control

- **Camacho & Bordons (2007)** - "Model Predictive Control"
  Fundamentos de MPC em sistemas de controle. A aplicacao de MPC para
  navegacao em manifold epistemico (MPC Navigator) e original do ATIC.

### Continual Learning

- **Kirkpatrick et al. (2017)** - "Overcoming catastrophic forgetting in neural networks" (EWC)
  Elastic Weight Consolidation via Fisher Information Matrix.

- **Chaudhry et al. (2019)** - "Tiny Episodic Memories in Continual Learning"
  Experience Replay com reservoir sampling.

### Bayesian Deep Learning

- **Blundell et al. (2015)** - "Weight Uncertainty in Neural Networks"
  Bayes by Backprop. O tau Bayesiano com prior InverseGamma no MAD e uma
  adaptacao original do ATIC.

### Meta-cognicao e Consciencia Artificial

- **Cleeremans et al. (2020)** - "Learning to be conscious"
  Self-model e meta-representacao. O SelfModel do ATIC estende com
  drives motivacionais (curiosidade, dominio, autonomia), que sao originais.

- **Tononi et al. (2016)** - "Integrated Information Theory (IIT)"
  Teoria da informacao integrada. O phi(M) do ATIC e inspirado no phi de
  Tononi mas com 4 componentes operacionais proprios (dimensional, dispersao,
  entropia, confianca), que sao uma contribuicao original.

### Conflito Phi-Psi

- **Frankfurt (1971)** - "Freedom of the Will and the Concept of a Person"
  Hierarquia de desejos (first-order/second-order). O Filosofia3 do ATIC
  operacionaliza este conceito como conflito phi-psi com deteccao por
  cosine similarity e mediacao negociada, que sao originais.

### Termodinamica Computacional

- **Landauer (1961)** - "Irreversibility and Heat Generation in the Computing Process"
  Principio de Landauer: apagar informacao tem custo termodinamico minimo.
  O modulo de termodinamica computacional do ATIC aplica este principio
  para medir custo de operacoes cognitivas irreversiveis, que e original.

### Contrastive Learning

- **Chen et al. (2020)** - "A Simple Framework for Contrastive Learning" (SimCLR)
  Aprendizado contrastivo. Adaptado no ATIC para anti-colapso metacognitivo
  com head contrastivo dual, que e original.

### Smooth Transition Penalty (STP)

- **Guo et al. (2025)** - "Straight-Through Meets Sparse Recovery" (arXiv:2602.22617)
  Propoe penalidade de transicao suave no espaco de hidden states.
  Para triplets (s, r, t) com s < r < t, a loss e:
  `L_stp = 1 - cos_sim(h_t - h_r, h_r - h_s)`.
  Encoraja trajetorias localmente lineares no espaco latente,
  melhorando suavidade de representacao e estabilidade de treinamento.
  Integrado no AletheionV2 como 14o componente da loss composta.

---

## Resumo

A distincao fundamental e:
- **Trabalhos academicos acima**: fornecem a teoria matematica e os frameworks gerais
- **Conceitos do ATIC (Felipe Maya Muniz)**: aplicam, estendem e combinam estas teorias
  de formas originais para criar o sistema epistemico completo
- **AletheionV2**: transpoe os conceitos do ATIC de Python para nn.Module treinaveis

Nenhum dos trabalhos academicos listados implementa DRM, VI, MOPsi, MPL, EidosDecay,
Filosofia3, Termodinamica Computacional, ou Tomografia Epistemica. Estes sao
contribuicoes originais de Felipe Maya Muniz.

---

(c) 2025-2026 Felipe Maya Muniz. All rights reserved.
