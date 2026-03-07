# 006 - Auditoria e Correcao dos Papers (PT + EN)

**Data:** 2026-03-07

## Contexto

Apos a criacao dos papers em PT e EN, foi feita uma auditoria completa
comparando o que o paper descreve vs o que o codigo implementa.
Encontradas 13 discrepancias criticas e 6 menores.

## Correcoes Aplicadas (ambos PT e EN)

### Secao 02 - Arquitetura
- FFN: SwiGLU -> GELU (2 camadas, nao 3)
- Embedding: adicionado scaling sqrt(d_model)

### Secao 03 - Sistema Epistemico
- Q1/Q2 gates: GELU+d/4 -> ReLU+hidden=64
- Temperatura adaptativa: MLP -> formula threshold (base/c se c<0.5)
- Tabela de tiers: params corrigidos (~49K por gate, ~16K manifold)

### Secao 04 - DRM Manifold
- Manifold MLP: 3 camadas d->32->16->5 -> 2 camadas d->20->5
- Adicionado sistema de 6 anchor points
- L_metric: (kappa-1)^2 -> log(kappa+1)

### Secao 05 - MAD Confianca
- Centroide: [0.5]*5 -> truth anchor [0.1,0.1,0.5,0.9,0.9]
- Adicionada distancia anisotropica (por eixo)
- tau^2 init: IG(2,1) prior -> exp(0)=1.0 com clamp [0.01, 10.0]

### Secao 06 - Vetor de Intencionalidade
- phi_1 (dim): indicador binario -> axis_std.mean()/0.29
- phi_2 (disp): media stds -> distancia L2 centroide/sqrt(5)
- phi_3 (ent): entropia campo direcional -> entropia coordenadas
- phi_4 (conf): normalizado por sigma* -> normalizado por 0.5
- Severidade: pura analitica -> blend 70/30 (analitica + MLP aprendido)
- Correcao: deterministica -> MLP aprendido (9->10->5, GELU+Tanh)
- VI loss: phi_min=0.3 -> phi_critical=0.5 + 0.3*severity^2

### Secao 07 - MPC Navigator
- Estado: (coords, kappa, phi) -> phi_components 4D
- 12 acoes corrigidas (INJECT_AXIS_0..4, INJECT_WEAKEST, INJECT_TWO,
  CONF_LIGHT/MEDIUM/STRONG, RESET_25)
- Transition model: 1 hidden ReLU -> 2 hidden GELU+Tanh com residual
- Funcao custo: 4 termos -> scoring por modo (RECOVERY/MAINTENANCE)

### Secao 08 - Modulos de Extensao
- CausalState gate: g=sigma(w^T[h;MLP(z)]) -> g=sigma(w^T z + b)

### Secao 09 - Loss Composta
- VARO: variancia condicional -> MSE(q1*q2, p_correct)
- lambda_varo: 0.01 -> 0.10
- lambda_mad: 0.01 -> 0.05
- L_metric: (kappa-1)^2 -> log(kappa+1)
- VI loss: phi_min=0.3 -> phi_critical=0.5 + severity

### Bibliografias
- Adicionada referencia GELU (Hendrycks & Gimpel, 2016)

## Arquivos Modificados

- paper/en/sections/02_architecture.tex
- paper/en/sections/03_epistemic_system.tex
- paper/en/sections/04_drm_manifold.tex
- paper/en/sections/05_mad_confidence.tex
- paper/en/sections/06_intentionality_vector.tex
- paper/en/sections/07_mpc_navigator.tex
- paper/en/sections/08_extension_modules.tex
- paper/en/sections/09_composite_loss.tex
- paper/en/references.bib
- paper/pt/sections/02_arquitetura.tex
- paper/pt/sections/03_sistema_epistemico.tex
- paper/pt/sections/04_manifold_drm.tex
- paper/pt/sections/05_mad_confianca.tex
- paper/pt/sections/06_vetor_intencionalidade.tex
- paper/pt/sections/07_mpc_navegador.tex
- paper/pt/sections/08_modulos_extensao.tex
- paper/pt/sections/09_loss_composta.tex
- paper/pt/references.bib

## Resultado

19 discrepancias corrigidas (13 criticas + 6 menores).
Papers agora refletem fielmente a implementacao do codigo.
