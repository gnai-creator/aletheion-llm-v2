# 009 - Real Geodesic: Campo Tensorial G(x) com Curvatura Real

Data: 2026-03-15
Branch: `real_geodesic` (derivada de `full_mahalanobis`)

## Contexto

A branch `main` e `full_mahalanobis` usam tensor metrico G constante (15 params Cholesky).
Isso define um espaco plano com metrica obliqua -- sem curvatura Riemanniana.

O paper propoe um manifold com curvatura real, o que exige G(x) -- tensor metrico
que varia com a posicao. Quando G depende da posicao, os simbolos de Christoffel
sao nao-zero e o tensor de Riemann e nao-trivial.

## O que foi implementado

### MetricNet (metric_tensor.py)
- MLP: Linear(5, 32) -> Tanh -> Linear(32, 15)
- Cholesky por ponto: G(x) = L(x) @ L(x)^T (SPD garantido)
- Zero init no ultimo layer -> G(x) ~ I no inicio (estabilidade)
- ~687 params extras (~0.0002% do modelo)

### Integral de Linha (metric_tensor.py)
- Quadratura Gauss-Legendre com 5 pontos
- Comprimento do segmento reto p->q sob metrica variavel:
  d(p,q) = integral_0^1 sqrt(delta^T G(gamma(t)) delta) dt
- Nao e geodesica verdadeira, mas captura variacao de G ao longo do caminho
- Diferenciavel end-to-end

### Distancia Geodesica (geodesic_distance.py)
- forward(): usa integral de linha quando metric_net disponivel
- batch_to_anchors(): usa G local (G avaliado nas coords do token)
- Fallback para Mahalanobis com G constante quando metric_net=None

### MADConfidence (confidence.py)
- Suporta G [D,D] constante e G [B,T,D,D] local
- Matmul adaptado para ambas dimensionalidades

### EpistemicHead (epistemic_head.py)
- Instancia MetricNet quando config.metric_position_dependent=True
- G_local = metric_net(coords) no forward
- metric_G armazenado em EpistemicTomography para uso na loss

### Regularizacao (composite_loss.py)
- _metric_reg_field(): condition proxy + scale penalty por ponto (vetorizado)
- Suavidade: perturba coords com ruido eps=0.01, penaliza variacao de G
- Parametro lambda_metric_smoothness (default 0.1)

### Trainer (trainer_distributed.py)
- LR separado para MetricNet (multiplicador configuravel, default 10x)
- Metricas: metric_G_var, metric_G_diag_mean, metric_G_frob_mean
- Migration de checkpoints antigos (strict=False + log de params faltantes)

### Output (output.py)
- Campo metric_G: Optional[Tensor] em EpistemicTomography

## Configuracao (config.py)

| Parametro | Default | Descricao |
|-----------|---------|-----------|
| metric_position_dependent | True | Habilita G(x) |
| metric_net_hidden | 32 | Hidden dim da MLP |
| metric_net_n_quad | 5 | Pontos Gauss-Legendre |
| metric_net_lr_multiplier | 10.0 | LR multiplicador |
| lambda_metric_smoothness | 0.1 | Peso suavidade |
| metric_max_condition | 50.0 | Condition number max |

## Arquivos Modificados

| Arquivo | Tipo |
|---------|------|
| src/aletheion_v2/config.py | 6 params novos |
| src/aletheion_v2/drm/metric_tensor.py | MetricNet novo (~130 linhas) |
| src/aletheion_v2/drm/geodesic_distance.py | Suporte metric_net |
| src/aletheion_v2/drm/__init__.py | Export MetricNet |
| src/aletheion_v2/mad/confidence.py | G batched |
| src/aletheion_v2/epistemic/epistemic_head.py | Wire MetricNet |
| src/aletheion_v2/core/output.py | Campo metric_G |
| src/aletheion_v2/loss/composite_loss.py | Reg field + smoothness |
| src/aletheion_v2/training/trainer_distributed.py | LR separado + migration |
| tests/test_drm.py | 13 testes novos |

## Testes

35/35 passam (pytest tests/test_drm.py).

## O que monitorar no treino

1. `metric_G_var`: variancia de G sobre o batch. Se ~0, G(x) e constante (espaco plano)
2. `metric_G_diag_mean`: media da diagonal de G (escala)
3. `metric_G_frob_mean`: norma Frobenius media (complexidade de G)
4. ECE/Brier por regiao epistemica (clusters de Q2 x complexidade)
5. Gradientes do MetricNet (monitorar se sinal e suficiente)

## Hipotese

Se G(x) convergir para funcao constante: espaco plano, full_mahalanobis era suficiente.
Se G(x) aprender estrutura local: curvatura real, calibracao pode melhorar por regiao.
