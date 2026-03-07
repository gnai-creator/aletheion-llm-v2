# Arquitetura do AletheionV2

## Visao Geral

O AletheionV2 e um LLM decoder-only com sistema epistemico intrinseco.
Diferente de LLMs tradicionais que produzem apenas logits, cada token
gera uma **tomografia epistemica** completa que inclui incerteza,
confianca, posicao no manifold, saude estrutural e estado cognitivo.

```
                           AletheionV2Model
                    +--------------------------+
                    |                          |
input_ids [B,T] -->| Embeddings               |
                    |   token_emb + RoPE       |
                    |                          |
                    | TransformerBlock x N      |
                    |   MultiHeadAttention     |
                    |   + RoPE rotary pos      |
                    |   + causal mask          |
                    |   FeedForward (SwiGLU)   |
                    |   LayerNorm (pre-norm)   |
                    |                          |
                    | ln_final                 |
                    |   |           |          |
                    | lm_head   EpistemicHead  |
                    | (tied)    (11 sub-heads) |
                    |   |           |          |
                    +--------------------------+
                        |           |
                    logits    EpistemicTomography
                   [B,T,V]     (30+ campos)
```

## Estrutura de Diretorios

```
src/aletheion_v2/
|
+-- config.py                    # AletheionV2Config (50+ campos)
|
+-- core/                        # Nucleo do modelo
|   +-- model.py                 # AletheionV2Model (forward principal)
|   +-- embeddings.py            # TokenEmbedding + RoPE
|   +-- transformer_block.py     # TransformerBlock (attn + ff + ln)
|   +-- output.py                # ModelOutput + EpistemicTomography
|
+-- epistemic/                   # Sistema epistemico central
|   +-- epistemic_head.py        # EpistemicHead (orquestra todos os sub-heads)
|   +-- gates.py                 # Q1Gate, Q2Gate, AdaptiveTemperature
|
+-- drm/                         # Differentiable Riemannian Manifold
|   +-- manifold_embedding.py    # Coords 5D [B,T,5] via MLP
|   +-- metric_tensor.py         # Tensor metrico SPD G [5,5]
|   +-- directional_field.py     # Campo direcional (entropia atencao)
|   +-- geodesic_distance.py     # Distancia geodesica via G
|
+-- mad/                         # Metric-Aware Distance
|   +-- bayesian_tau.py          # Tau^2 Bayesiano (InverseGamma prior)
|   +-- confidence.py            # Confianca MAD Gaussiana
|
+-- vi/                          # Vetor de Intencionalidade
|   +-- phi_field.py             # phi(M) = sum(w_i * phi_i) 4 componentes
|   +-- intentionality_vector.py # Severidade + correcao de confianca
|
+-- mpc/                         # Model Predictive Control
|   +-- transition_model.py      # Modelo de transicao (12 acoes)
|   +-- navigator.py             # ManifoldNavigator (beam search K=4, D=3)
|
+-- eidos/                       # Tier 1 - Balanceamento estrutural
|   +-- eidos_decay.py           # EidosDecay (~368 params)
|
+-- filosofia3/                  # Tier 1 - Conflito phi-psi
|   +-- conflict_head.py         # PhiPsiConflictHead (~373 params)
|
+-- consciousness/               # Tier 1 - Auto-modelo
|   +-- self_model_head.py       # SelfModelHead (~982 params)
|
+-- grounding/                   # Tier 2 - Ancoragem
|   +-- task_head.py             # TaskClassificationHead (~50K params)
|   +-- ambiguity_head.py        # AmbiguityHead (~50K params)
|
+-- plasticity/                  # Tier 2 - Plasticidade sinaptica
|   +-- plasticity_gate.py       # PlasticityGate (~12K params)
|
+-- mpl/                         # Tier 2 - Projecao de longo prazo
|   +-- density_tracker.py       # DensityTracker (hash-grid 5D)
|   +-- frontier_head.py         # FrontierHead (~129 params)
|
+-- mopsi/                       # Tier 3 - Orientacao psi
|   +-- human_state_head.py      # HumanStateHead + PhiPsiMediator (~25K)
|
+-- causal_state/                # Tier 3 - Estado causal
|   +-- state_conditioning.py    # StateConditioning + PolicyBinding (~26K)
|
+-- metacognitive/               # Tier 3 - Meta-cognicao
|   +-- contrastive_head.py      # ContrastiveHead (~200K params)
|
+-- loss/                        # 13 funcoes de perda
|   +-- composite_loss.py        # Agregador principal com annealing
|   +-- varo_loss.py             # Variancia condicional Q1*Q2
|   +-- vi_regularization.py     # Regularizacao VI (phi minimo)
|   +-- mad_calibration.py       # Calibracao MAD (confianca vs acuracia)
|   +-- eidos_loss.py            # Balanceamento de eixos
|   +-- conflict_loss.py         # Penaliza conflito phi-psi
|   +-- consciousness_loss.py    # Energia minima
|   +-- grounding_loss.py        # Entropia + calibracao ambiguidade
|   +-- plasticity_loss.py       # Plasticidade minima
|   +-- frontier_loss.py         # Exploracao de fronteira
|   +-- mopsi_loss.py            # Alinhamento psi-confianca
|   +-- contrastive_loss.py      # Anti-colapso contrastivo
|
+-- training/                    # Pipeline de treinamento
|   +-- trainer.py               # Trainer single-GPU
|   +-- trainer_distributed.py   # DistributedTrainer (DDP/FSDP)
|   +-- distributed.py           # Setup DDP/FSDP + mixed precision
|   +-- data.py                  # Datasets sinteticos
|   +-- data_pipeline.py         # MemmapDataset, streaming HF, mixing
|   +-- scheduler.py             # WarmupCosine + LossWeightAnnealer
|   +-- ewc.py                   # EWC (Continual Learning)
|   +-- replay_buffer.py         # Experience Replay (Continual Learning)
|
+-- inference/                   # Geracao
|   +-- generator.py             # Generator (top-k, MPC opcional)
|   +-- dashboard_bridge.py      # Ponte para dashboard ATIC
|
+-- tokenizer/                   # Tokenizacao
    +-- tokenizer.py             # tiktoken / sentencepiece wrapper
```

## Fluxo de Forward Pass

### 1. Embeddings

```python
# input_ids [B, T] -> hidden_states [B, T, d_model]
token_emb = self.token_emb(input_ids)      # Lookup table
pos_info = RoPE(positions)                  # Rotary Position Encoding
hidden = token_emb                          # RoPE aplicado dentro da atencao
```

### 2. Transformer Blocks (x N)

```python
for block in self.blocks:
    # Pre-norm (estilo GPT-2/Llama)
    normed = block.ln1(hidden)
    attn_out, attn_weights = block.attention(normed, pos_info)
    hidden = hidden + attn_out              # Residual

    normed = block.ln2(hidden)
    ff_out = block.feed_forward(normed)     # SwiGLU: gate * up * down
    hidden = hidden + ff_out                # Residual
```

### 3. EpistemicHead

O EpistemicHead recebe `hidden_states [B,T,d_model]` e `attention_patterns [B,L,H,T,T]`
e produz a tomografia completa em 3 tiers:

```python
# --- Core ---
q1 = self.q1_gate(hidden)                     # Incerteza aleatoria [B,T,1]
q2 = self.q2_gate(hidden)                     # Incerteza epistemica [B,T,1]
coords = self.manifold_embedding(hidden)       # DRM coords [B,T,5]
G = self.metric_tensor()                       # Tensor metrico [5,5]
distance = self.geodesic_distance(coords, G)   # Distancia [B,T,1]
confidence = self.mad_confidence(coords)       # MAD confidence [B,T,1]
phi = self.phi_field(coords, confidence)       # phi(M) [B,T,1]
vi = self.intentionality(phi, confidence)      # Correcao VI [B,T,5]
tau = self.adaptive_temp(q1, q2)               # Temperatura [B,T,1]

# --- Tier 1 (~1.7K params) ---
eidos = self.eidos_decay(coords, confidence)   # axis_balance [B,T,5]
conflict = self.conflict_head(phi_comp, conf)  # conflict_intensity [B,T,1]
self_model = self.self_model(hidden, q2, phi)  # mood, energy, drives

# --- Tier 2 (~112K params) ---
task = self.task_head(hidden)                  # task_probs [B,T,9]
ambiguity = self.ambiguity_head(hidden)        # ambiguity_level [B,T,1]
plasticity = self.plasticity_gate(hidden, vi)  # plasticity [B,T,1]
frontier = self.frontier_head(coords, density) # frontier_score [B,T,1]

# --- Tier 3 (~250K params) ---
human = self.human_state(hidden)               # human_state_5d [B,T,5]
mediator = self.mediator(phi, psi, conflict)   # mediation [B,T,1]
contrastive = self.contrastive(hidden)         # divergence [B,T,1]
```

## Loss Composta

13 componentes com annealing:

```
L_total = lambda_ce * CE
        + anneal * (
            lambda_varo * VARO          # Q1*Q2 variance
          + lambda_vi * VI_reg          # phi minimo
          + lambda_mad * MAD_cal        # confianca calibrada
          + lambda_metric * metric_reg  # condition number de G
          + lambda_eidos * eidos_reg    # balanceamento eixos
          + lambda_conflict * conflict  # minimiza conflito
          + lambda_consciousness * cons # energia minima
          + lambda_grounding * ground   # entropia + calibracao
          + lambda_plasticity * plast   # plasticidade minima
          + lambda_frontier * frontier  # maximiza exploracao
          + lambda_mopsi * mopsi        # alinhamento psi
          + lambda_contrastive * contr  # anti-colapso
        )
```

**Annealing schedule:**
- Steps 0 - 10%: `anneal = 0` (so CE)
- Steps 10% - 50%: `anneal` cresce linearmente de 0 a 1
- Steps 50% - 100%: `anneal = 1` (todas as losses ativas)

## Manifold DRM 5D

O modelo opera num manifold Riemanniano 5D onde cada token e mapeado:

```
hidden [B,T,768] -> MLP -> coords [B,T,5] in [0,1]^5
```

As 5 dimensoes capturam diferentes aspectos epistemicos:
- **Dim 0**: Certeza factual
- **Dim 1**: Consistencia logica
- **Dim 2**: Contexto/grounding
- **Dim 3**: Calibracao de confianca
- **Dim 4**: Profundidade epistemica

O tensor metrico G[5,5] e SPD (positivo-definido) e define a geometria:
- Distancias geodesicas entre tokens
- Curvatura de Ricci (saude do manifold)
- Direcoes preferenciais de exploracao

## Continual Learning

### EWC (Elastic Weight Consolidation)

Apos cada fase de treinamento, computa a Fisher Information Matrix diagonal:

```
F_i = E[ (dL/d_theta_i)^2 ]    # Importancia do parametro i
```

Na fase seguinte, adiciona penalidade:

```
L_ewc = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
```

Isso impede que parametros importantes para tarefas anteriores mudem muito.

### Experience Replay

Reservoir sampling mantem buffer de amostras anteriores em CPU.
Durante treinamento, substitui `mix_ratio` (default 10%) do batch atual
com amostras do buffer.

## Scaling (1M - 640B)

| Escala | d_model | n_heads | n_layers | d_ff |
|--------|---------|---------|----------|------|
| 1M | 64 | 2 | 4 | 256 |
| 10M | 256 | 4 | 6 | 1024 |
| 50M | 512 | 8 | 8 | 2048 |
| 125M | 768 | 12 | 12 | 3072 |
| 350M | 1024 | 16 | 24 | 4096 |
| 1.3B | 2048 | 16 | 24 | 8192 |
| 7B | 4096 | 32 | 32 | 11008 |
| 13B | 5120 | 40 | 40 | 13824 |
| 30B | 6656 | 52 | 60 | 17920 |
| 70B | 8192 | 64 | 80 | 28672 |
| 162B | 12288 | 96 | 96 | 49152 |
| 250B | 16384 | 128 | 80 | 65536 |
| 400B | 18432 | 144 | 96 | 73728 |
| 640B | 20480 | 160 | 128 | 81920 |

**Padrao de escalamento:**
- `head_dim` = 32 (1M), 64 (10M-350M), 128 (1.3B+)
- `d_ff` = 4 * d_model (standard) ou SwiGLU (7B+)
- Gradient checkpointing a partir de 1.3B
- FSDP a partir de 7B
- Dropout 0.0 a partir de 7B

## Pipeline de Treinamento

```
                    AletheionV2Config (YAML)
                           |
                    +------+------+
                    |             |
              AletheionV2Model  DataLoader
                    |             |
                    +------+------+
                           |
                  DistributedTrainer
                    |             |
              +-----+-----+ +----+----+
              |           | |         |
           DDP/FSDP   Mixed   Gradient  WarmupCosine
                      Prec   Checkpt   Scheduler
                           |
                    AletheionV2Loss
                    (13 componentes)
                           |
                    +------+------+
                    |             |
                   EWC        Replay
                  (Fisher)   (Buffer)
                           |
                    Checkpoint
                    (model + optim + ewc + replay)
```

## Testes

261 testes cobrindo:

| Suite | Testes | Cobertura |
|-------|--------|-----------|
| test_model | 7 | Forward, backward, causal mask, weight tying |
| test_epistemic | 12 | Q1/Q2 gates, temperatura, EpistemicHead |
| test_drm | 8 | Manifold, metrica, geodesicas |
| test_vi | 13 | Phi field, VI, Bayesian tau, MAD |
| test_eidos | 10 | EidosDecay + loss |
| test_filosofia3 | 10 | Conflito phi-psi + loss |
| test_consciousness | 10 | SelfModel + loss |
| test_grounding | 15 | TaskHead + AmbiguityHead + loss |
| test_plasticity | 10 | PlasticityGate + loss |
| test_mpl | 15 | DensityTracker + FrontierHead + loss |
| test_mopsi | 12 | HumanState + Mediator + loss |
| test_causal_state | 10 | StateConditioning + PolicyBinding |
| test_metacognitive | 12 | ContrastiveHead + loss |
| test_continual_learning | 51 | EWC + Replay + configs |
| test_integration | 15 | End-to-end + MPC + data pipeline |

```bash
pytest tests/ -v                     # Todos
pytest tests/test_model.py -v        # So modelo core
pytest tests/test_continual_learning.py -v  # So CL
```
