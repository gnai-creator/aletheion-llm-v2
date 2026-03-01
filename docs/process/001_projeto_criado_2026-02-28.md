# 001 - Criacao do aletheion-llm-v2

**Data:** 2026-02-28
**Autor:** Claude (assistido)

## Resumo

Criado repositorio `aletheion-llm-v2` com reescrita completa do aletheion-llm.
LLM com DRM/MAD/VI/MPC integrados como `nn.Module` treinaveis no core,
produzindo tomografia epistemica por token.

## Estrutura

- 36 arquivos Python fonte (3,888 LOC)
- 5 arquivos de teste (973 LOC)
- Total: ~4,861 LOC
- Todos os arquivos < 500 linhas (modularidade respeitada)

## Modulos Implementados

### Core
- `config.py` - AletheionV2Config (dataclass, YAML I/O)
- `embeddings.py` - Token embedding + RoPE
- `transformer_block.py` - Pre-LN block com retorno de attention patterns
- `model.py` - AletheionV2Model (forward + generate_next_token)
- `output.py` - EpistemicTomography, ModelOutput

### Epistemic
- `gates.py` - Q1Gate, Q2Gate, AdaptiveTemperature (portado de epistemic_softmax)
- `epistemic_head.py` - Orquestrador DRM+MAD+VI -> EpistemicTomography

### DRM (Directional Relational Manifold)
- `manifold_embedding.py` - hidden -> coords 5D + anchor distances
- `metric_tensor.py` - G = L@L^T (Cholesky SPD, sempre positiva definida)
- `directional_field.py` - attn entropy -> directions + dim_D
- `geodesic_distance.py` - Mahalanobis ao truth centroid

### MAD (Metric-Aware Decay)
- `bayesian_tau.py` - tau^2 aprendivel por eixo via exp(log_tau_sq)
- `confidence.py` - C(p) = exp(-d^2/2tau^2)

### VI (Vetor de Intencionalidade)
- `phi_field.py` - phi(M) = w*phi_dim + w*phi_disp + w*phi_ent + w*phi_conf
- `intentionality_vector.py` - direction + severity (homeostase)

### MPC (Model Predictive Control)
- `transition_model.py` - T(s,a)->s' (analitico + residuo neural)
- `navigator.py` - Beam search K=4, D=3

### Loss
- `varo_loss.py` - Calibracao Q1*Q2 vs acuracia real
- `vi_regularization.py` - Penaliza phi baixo + severity alta
- `mad_calibration.py` - BCE confidence vs acuracia
- `composite_loss.py` - CE + VARO + VI + MAD + metric_reg com annealing

### Training
- `data.py` - TextChunkDataset + DataLoader factory
- `scheduler.py` - Warmup + cosine decay + loss weight annealing
- `trainer.py` - Loop completo de treinamento

### Inference
- `generator.py` - Geracao auto-regressiva com tomografia
- `dashboard_bridge.py` - Conversao para formato ATIC dashboard

## Testes

85 testes, 85 passando:
- test_drm.py (21 testes): anchors, manifold embedding, metric tensor, directional field, geodesic
- test_epistemic.py (16 testes): gates, temperatura, EpistemicHead completo
- test_vi.py (18 testes): PhiField, IntentionalityVector, BayesianTau, MADConfidence
- test_model.py (14 testes): config, model, loss composta
- test_integration.py (16 testes): E2E, MPC, data pipeline, dashboard bridge

## Decisoes Chave

| Decisao | Escolha |
|---------|---------|
| DRM coords | Linear + Sigmoid em [0,1] |
| Metric tensor | Cholesky L@L^T + eps*I (garante SPD) |
| MAD tau | nn.Parameter(log_tau_sq), positividade via exp |
| VI phi | Batch statistics (diferenciavel, sem historico) |
| MPC | Inferencia-only (beam search, nao diferenciavel) |
| Weight tying | lm_head.weight = token_emb.weight |

## Referencia

Portado de:
- `epistemic_softmax/epistemic_softmax.py` (Q1/Q2 gates)
- `atic_consulting/atic/src/core/drm/` (DRM/MAD/VI/MPC em NumPy)
