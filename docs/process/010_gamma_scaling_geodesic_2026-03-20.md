# 010 - Gamma-Scaling Relativistic na Distancia Geodesica

Data: 2026-03-20

## Contexto

No paper DRM Relativistic Dynamics (Proposicao 4.2), definimos o fator de Lorentz:

```
gamma(v) = 1 / sqrt(1 - v^2/c^2)
```

Onde `v` e a distancia euclidiana ao anchor mais proximo e `c = sqrt(5)` (diagonal maxima em [0,1]^5).

## Motivacao

A distancia geodesica tratava todo o manifold com a mesma resolucao. Regioes longe dos anchors (alta incerteza) precisam de maior discriminacao metrica. O gamma(v) escala a distancia: ~1.0 perto dos anchors, cresce suavemente longe deles.

## O que foi implementado

### 1. `src/aletheion_v2/drm/geodesic_distance.py`

- Funcao `_gamma_scale(coords, anchors, c_param, eps)`: computa fator gamma [B, T, 1]
  - Calcula distancia euclidiana a cada anchor
  - Usa minimo (anchor mais proximo)
  - Aplica formula relativistic gamma(v)
  - Clamp v < 0.999*c para seguranca numerica

- Classe `GeodesicDistance`: novos parametros no construtor:
  - `gamma_enabled: bool = False` (zero impacto se desativado)
  - `gamma_c_param: float = sqrt(5)`

- Metodos modificados:
  - `forward()`: aceita `anchors` opcional, aplica gamma na distancia final
  - `pairwise()`: aceita `anchors`, usa midpoint dos dois pontos para gamma
  - `batch_to_anchors()`: aplica gamma por token (broadcast [B,T,1] -> [B,T,A])

### 2. `src/aletheion_v2/config.py`

- `gamma_scaling_enabled: bool = False` (default off)
- `gamma_c_param: float = 2.236` (sqrt(5))

### 3. `src/aletheion_v2/epistemic/epistemic_head.py`

- `GeodesicDistance` instanciado com gamma config
- Forward passa `anchors=self.manifold_emb.anchors` para geodesic_dist

### 4. `configs/scaling/50m_real_geodesic_rtx4090.yaml`

- `gamma_scaling_enabled: true`
- `gamma_c_param: 2.236`

## Comportamento Verificado

| Posicao | Dist min anchor | Gamma |
|---------|----------------|-------|
| Perto truth | 0.12 | 1.001 |
| Centro [0.5]^5 | 0.40 | 1.016 |
| Extremo [0]^5 | 0.77 | 1.066 |
| Corner [1,1,0,0,0] | 1.02 | 1.124 |

O efeito e suave — nao altera dramaticamente as distancias, mas adiciona 6-12% de boost em regioes extremas.

## Arquivos Modificados

- `src/aletheion_v2/drm/geodesic_distance.py` (reescrito)
- `src/aletheion_v2/config.py` (2 campos adicionados)
- `src/aletheion_v2/epistemic/epistemic_head.py` (2 edits)
- `configs/scaling/50m_real_geodesic_rtx4090.yaml` (2 linhas)

## Tambem nesta sessao

- Implementado `lambda_metric_diversity` loss term (ver composite_loss.py)
- Fix: optimizer migration quando param groups diferem (trainer_distributed.py)
