# Arte Anterior e Trabalhos Relacionados

Este documento lista os trabalhos que influenciaram ou fundamentam
a arquitetura do Aletheion LLM v2.

## Arquitetura Transformer

- **Vaswani et al. (2017)** - "Attention Is All You Need"
  Arquitetura Transformer original com multi-head self-attention.

- **Radford et al. (2019)** - "Language Models are Unsupervised Multitask Learners" (GPT-2)
  Decoder-only com pre-norm residual. Base arquitetural do Aletheion.

- **Touvron et al. (2023)** - "LLaMA: Open and Efficient Foundation Language Models"
  RoPE, SwiGLU feed-forward, pre-norm. Padroes adotados no Aletheion.

## Incerteza Epistemica

- **Der Kiureghian & Ditlevsen (2009)** - "Aleatory or epistemic? Does it matter?"
  Distincao formal entre incerteza aleatoria (Q1) e epistemica (Q2).

- **Gal & Ghahramani (2016)** - "Dropout as a Bayesian Approximation"
  MC Dropout para estimacao de incerteza. Inspiracao para gates Q1/Q2.

- **Lakshminarayanan et al. (2017)** - "Simple and Scalable Predictive Uncertainty Estimation"
  Deep ensembles. Motivacao para abordagem alternativa via manifold.

## Geometria Riemanniana em ML

- **Bronstein et al. (2017)** - "Geometric Deep Learning"
  Framework para deep learning em dominios nao-Euclidianos.

- **Nickel & Kiela (2017)** - "Poincare Embeddings for Learning Hierarchical Representations"
  Embeddings em espacos hiperbolicos. Inspiracao para manifold 5D.

## Model Predictive Control

- **Camacho & Bordons (2007)** - "Model Predictive Control"
  Fundamentos de MPC. Adaptado para navegacao no manifold epistemico.

## Continual Learning

- **Kirkpatrick et al. (2017)** - "Overcoming catastrophic forgetting in neural networks" (EWC)
  Elastic Weight Consolidation via Fisher Information Matrix.

- **Chaudhry et al. (2019)** - "Tiny Episodic Memories in Continual Learning"
  Experience Replay com reservoir sampling.

## Bayesian Deep Learning

- **Blundell et al. (2015)** - "Weight Uncertainty in Neural Networks"
  Bayes by Backprop. Inspiracao para tau Bayesiano com prior InverseGamma.

## Meta-cognicao e Consciencia Artificial

- **Cleeremans et al. (2020)** - "Learning to be conscious"
  Self-model e meta-representacao. Base para SelfModelHead.

- **Tononi et al. (2016)** - "Integrated Information Theory (IIT)"
  Teoria da informacao integrada. Influencia no phi(M).

## Conflito Phi-Psi

- **Frankfurt (1971)** - "Freedom of the Will and the Concept of a Person"
  Hierarquia de desejos (first-order/second-order). Base para Filosofia3.

## Contrastive Learning

- **Chen et al. (2020)** - "A Simple Framework for Contrastive Learning" (SimCLR)
  Aprendizado contrastivo. Adaptado para anti-colapso metacognitivo.

---

Este documento nao e exaustivo. Para fundamentacao teorica completa,
consulte os papers em `paper/en/` e `paper/pt/`.
