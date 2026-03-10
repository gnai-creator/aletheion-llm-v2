# AletheionV2 Training Log Variables

Complete reference for all variables logged during training.
Each variable is explained with an **analogy** to make the concept intuitive.

---

## Training Control

| Variable | Description | Analogy |
|----------|-------------|---------|
| `step` | Current training step (batch iteration) | Odometer of the car |
| `epoch` | Current epoch (full pass over dataset) | How many laps around the track |
| `tokens_seen` | Total tokens processed so far | Total kilometers driven |
| `tokens_per_sec` | Training throughput | Speed of the car (km/h) |
| `lr` | Current learning rate | Pressure on the gas pedal -- starts gentle, then firm, then eases off |
| `grad_norm` | L2 norm of all gradients | How steep the hill the car is climbing -- too steep risks sliding |
| `annealing` | Epistemic loss ramp factor [0, 1] | Volume knob for epistemic losses -- starts at 0 (silent) and slowly turns to full (1.0) during 30%-80% of training |

---

## Loss Components

The total loss is the sum of CE + 13 epistemic components. Think of it as a **dashboard with multiple gauges** -- CE keeps the engine running (language quality), while the others tune handling, safety, and awareness.

| Variable | Full Name | Description | Analogy |
|----------|-----------|-------------|---------|
| `total` | Total Loss | Weighted sum of all losses | Total fuel consumption |
| `ce` | Cross-Entropy | Standard language modeling loss -- predicting next token | The engine itself. Lower = better at predicting words |
| `stp` | Smooth Transition Penalty | Penalizes jagged trajectories in hidden states between consecutive tokens | Suspension smoothness. Penalizes sharp bumps in the road of representations |
| `varo` | Variational Reliability Optimization | Forces q1*q2 to match actual prediction accuracy | Speedometer calibration. Makes sure the confidence gauge matches real performance |
| `vi` | VI Regularization | Penalizes low phi and high severity | Check engine light penalty. When manifold health drops below critical, this kicks in |
| `mad` | MAD Calibration | Binary cross-entropy between confidence and correctness | Brake calibration. Makes sure confidence maps to actual correctness |
| `metric` | Metric Regularization | Prevents the metric tensor from becoming lopsided | Wheel alignment. Keeps all dimensions of the geometry balanced |
| `eidos` | Eidos Regularization | Regularizes axis balance quality | Tire pressure balancer. Ensures no single manifold axis dominates |
| `conflict` | Conflict Regularization | Regularizes phi-psi conflict intensity | Argument mediator. Prevents the model's health vs human satisfaction from diverging too much |
| `consciousness` | Consciousness Regularization | Regularizes energy, mood, and drives | Mental health check. Keeps the self-model's internal state reasonable |
| `grounding` | Grounding Regularization | Regularizes task confidence and ambiguity detection | GPS calibration. Ensures the model knows what kind of task it's doing |
| `plasticity` | Plasticity Regularization | Regularizes plasticity gate behavior | Battery management. Prevents learning capacity from draining or saturating |
| `frontier` | Frontier Regularization | Regularizes exploration scoring | Radar calibration. Tunes the novelty detector |
| `mopsi` | MOPsi Regularization | Regularizes human-model alignment | Empathy tuning. Keeps the model's sense of human satisfaction grounded |
| `contrastive` | Contrastive Regularization | Regularizes representation consistency | Mirror alignment. Two views of the same thing should agree |

### Lambda Decay

All epistemic loss weights decay exponentially: `lambda(t) = lambda_0 * exp(-k * t)`.
With `k=0.0001`, lambdas retain ~61% at step 5K and ~25% at the end. This is like **training wheels that slowly retract** -- epistemic structure is built early, then the model runs on its own.

---

## Core Epistemic Metrics (DRM)

The **DRM (Directional Relational Manifold)** is a 5-dimensional space where each token lives. Think of it as a **GPS for knowledge** -- every token has coordinates that describe its epistemic state.

### Uncertainty Gates

| Variable | Description | Analogy |
|----------|-------------|---------|
| `avg_q1` | Aleatoric uncertainty -- irreducible data noise | Fog density. How inherently unclear the input is, regardless of how smart the model is |
| `avg_q2` | Epistemic uncertainty -- model knowledge gaps | The model's "I don't know" meter. High = model needs more training on this topic |

### DRM Coordinates (5D Manifold)

Each token is placed in a 5-dimensional epistemic space. Think of a **5-axis radar chart** where each axis captures a different aspect of knowledge state.

| Variable | Axis | Analogy |
|----------|------|---------|
| `drm_coord_0` | Aleatoric dimension | How foggy the road is |
| `drm_coord_1` | Epistemic dimension | How well the driver knows the route |
| `drm_coord_2` | Domain complexity | How winding the road is |
| `drm_coord_3` | Temporal relevance | How current the road map is |
| `drm_coord_4` | Response quality | How well-paved the road is |
| `drm_coord_std` | Spread of coordinates across axes | How varied the radar shape is -- low means all axes similar, high means specialized |

### Axis Balance

| Variable | Description | Analogy |
|----------|-------------|---------|
| `axis_balance_0..4` | Per-axis reinforcement/decay factor | Tire pressure per wheel. Values >1.0 mean that axis is being inflated (under-represented), <1.0 means deflated (over-represented). Target: all near 1.0 |

---

## Manifold Health (Phi)

**Phi** is the overall health of the epistemic manifold. Think of it as the **wellness score of the model's inner knowledge map**. If phi drops, the model is losing its grip on what it knows and doesn't know.

| Variable | Description | Weight | Analogy |
|----------|-------------|--------|---------|
| `avg_phi` | Total manifold health score [0, 1] | -- | Overall fitness score of the athlete |
| `phi_dim` | Dimensional diversity | 35% | Flexibility -- can the athlete stretch in all directions? |
| `phi_disp` | Dispersion from centroid | 25% | Exploration range -- how far from home base the athlete ventures |
| `phi_ent` | Entropy across axes | 25% | Balance -- is the athlete equally trained in all skills, or lopsided? |
| `phi_conf` | Confidence variance quality | 15% | Emotional stability -- confidence should vary moderately (std ~0.15), not be flat or erratic |

---

## Confidence & Calibration

| Variable | Description | Analogy |
|----------|-------------|---------|
| `avg_confidence` | MAD confidence score [0, 1] | How sure the model is about each token. Like a doctor's confidence in a diagnosis |
| `avg_temperature` | Adaptive softmax temperature | Creativity dial. High temperature = more exploratory/creative, low = more deterministic. Scales inversely with certainty |
| `avg_vi_severity` | VI correction severity [0, 1] | Emergency level. 0 = healthy, 1 = critical. Triggers when phi drops below 0.5 |
| `avg_geodesic_distance` | Distance to truth centroid on manifold | How far each token is from "truth" on the knowledge map. Like distance from a lighthouse |

---

## Self-Model (Consciousness Module)

The model has a rudimentary **self-model** -- not sentience, but a computational analogue of introspective monitoring. Think of it as a **cockpit instrument panel** that monitors the model's own state.

| Variable | Description | Analogy |
|----------|-------------|---------|
| `avg_mood` | Computational mood [-1, 1] | Barometer of computational well-being. High confidence + high phi = good mood |
| `avg_curiosity` | Epistemic curiosity [0, 1] | Directly from q2 (epistemic uncertainty). High uncertainty = high curiosity. Like a student asking "what's that?" |
| `avg_energy` | Computational energy budget [0, 1] | Battery level. Decays along the sequence (30% loss end-to-end). Early tokens get more "attention budget" |
| `drive_autonomy` | Efficiency drive [0, 1] | Self-sufficiency instinct. How much the model wants to handle things on its own |
| `drive_curiosity` | Exploration drive [0, 1] | Curiosity instinct. How much the model wants to learn new things |
| `drive_mastery` | Goal completion drive [0, 1] | Achievement instinct. How much the model wants to get the answer right |

---

## Conflict Detection (Filosofia3)

The model monitors **tension between its own health (phi) and estimated human satisfaction (psi)**. Think of a **doctor who must balance medical correctness with patient preferences**.

| Variable | Description | Analogy |
|----------|-------------|---------|
| `avg_conflict` | Phi-psi conflict intensity [0, 1] | Tension between "what's medically correct" and "what the patient wants to hear" |
| `mode_prob_0` | P(ALIGNED) | Probability: doctor and patient agree |
| `mode_prob_1` | P(CONFLICT_TOLERATED) | Probability: minor disagreement, tolerable |
| `mode_prob_2` | P(SIGNAL_HUMAN) | Probability: significant enough to flag |
| `mode_prob_3` | P(RECOVERY) | Probability: severe conflict, needs intervention |

---

## Grounding & Context

| Variable | Description | Analogy |
|----------|-------------|---------|
| `avg_task_confidence` | Confidence in task type classification | How sure a receptionist is about which department to route you to |
| `avg_ambiguity` | Token ambiguity level [0, 1] | How many valid interpretations a word/phrase has. "Bank" near a river vs a financial district |

---

## Human Interaction (MOPsi)

| Variable | Description | Analogy |
|----------|-------------|---------|
| `avg_psi` | Estimated human satisfaction [0, 1] | The model's guess at how satisfied the human will be. Like a waiter gauging if the dish will please the customer |
| `avg_mediated` | Phi-psi mediated score [0, 1] | Compromise score. In high conflict, leans toward model integrity (phi). In low conflict, balances both |

---

## Exploration & Adaptation

| Variable | Description | Analogy |
|----------|-------------|---------|
| `avg_frontier` | Frontier/novelty score [0, 1] | How "unexplored" a region of the manifold is. Like a map showing uncharted territory. High = novel, low = well-mapped |
| `avg_plasticity` | Remaining learning capacity [0, 1] | Brain plasticity. Starts high (young brain, eager to learn), depletes over the sequence. Like ink in a pen |
| `avg_gate` | Plasticity gate value [0, 1] | The valve controlling how much modification is allowed. Low plasticity + high severity = gate opens for emergency learning |
| `avg_divergence` | Representation consistency [0, 1] | Two independent projections of the same hidden state. Low divergence = robust representation. Like asking two witnesses -- do their stories match? |
| `avg_eidos_weight` | Eidos modulation weight [0, 1] | How much the axis-balancer trusts the current confidence. Low when axes are imbalanced |
| `avg_dim_d` | Directional dimensionality [1, 5] | How many attention directions are "active". 1 = all heads look the same way, 5 = fully diverse. Like the number of spotlights pointing in different directions |

---

## Training Progression Summary

During the v2 epistemic fine-tuning (1B tokens, 5xH200 fp32):

| Phase | Steps | What Happens |
|-------|-------|-------------|
| **Warmup** (0-30%) | 0-4185 | Only CE + STP active. Model relearns language basics under fp32. phi starts low (~0.42) |
| **Ramp** (30-80%) | 4185-11160 | Epistemic losses ramp linearly. phi climbs rapidly (0.42 -> 0.66). 16 SKIP-GRADs occur in this zone |
| **Full** (80-100%) | 11160-13950 | All losses at full strength + lambda decay. phi stabilizes ~0.67. Model refines calibration |

### Key Metric Evolution

| Metric | Start (step 50) | End (step 13950) | Interpretation |
|--------|-----------------|-------------------|---------------|
| CE | 3.23 | 3.29 | Slight increase -- small price for epistemic capability |
| phi | 0.418 | 0.676 | +62% -- manifold health dramatically improved |
| confidence | 0.589 | 0.401 | -32% -- model became more calibrated, less overconfident |
| temperature | 3.97 | 9.78 | +146% -- model learned to be more exploratory when uncertain |
| vi_severity | 0.433 | 0.120 | -72% -- manifold rarely needs emergency correction anymore |
| frontier | 1.97 | 0.79 | -60% -- manifold has been well-explored, less uncharted territory |
| contrastive | 0.690 | 0.004 | -99% -- representations became extremely consistent |
| grounding | 1.078 | 0.009 | -99% -- task classification became confident |

---

## Eval Metrics (WikiText-103 OOD)

| Metric | What It Measures | Analogy |
|--------|-----------------|---------|
| PPL (Perplexity) | Language modeling quality | How "surprised" the model is by the text. Lower = better predictions |
| ECE (Expected Calibration Error) | Average gap between confidence and accuracy | If you say "80% sure" on 100 answers, you should get ~80 right. ECE measures the average gap |
| MCE (Maximum Calibration Error) | Worst-case calibration gap | The single worst confidence bin. Like finding the worst-calibrated thermometer |
| Brier Score | Overall probabilistic accuracy | Mean squared error between predictions and outcomes. Like the distance between darts and bullseye |
| Overconfidence Ratio | % tokens where confidence > accuracy | How often the model is cockier than it should be |

### v2 Fine-tuning Results

| Metric | Backbone | Fine-tuned v2 | Delta |
|--------|----------|--------------|-------|
| PPL | 46.90 | 49.42 | +5.4% (small cost) |
| **ECE** | 0.0284 | **0.0176** | **-38.0%** |
| **MCE** | 0.0766 | **0.0521** | **-32.0%** |
| **Brier** | 0.1536 | **0.1528** | **-0.5%** |
| **Overconf** | 92.55% | **77.29%** | **-16.5%** |

The fine-tuned model achieved the **best ECE (0.0176) and best Brier (0.1528)** among all tested models, including GPT-2 Medium and OPT-350M.
