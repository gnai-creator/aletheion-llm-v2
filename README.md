# Aletheion LLM v2

LLM com DRM/MAD/VI/MPC integrados como `nn.Module` treinaveis.
Cada token produz **tomografia epistemica** completa no manifold 5D.

## Arquitetura

```
input_ids -> Embeddings -> TransformerBlock x N -> ln_final
                                                    |
                                         +----------+-----------+
                                         |                      |
                                     lm_head               EpistemicHead
                                         |                      |
                                     logits             EpistemicTomography
```

### EpistemicTomography (por token)

| Campo | Shape | Descricao |
|-------|-------|-----------|
| q1 | [B,T,1] | Incerteza aleatoria |
| q2 | [B,T,1] | Incerteza epistemica |
| confidence | [B,T,1] | MAD confidence |
| drm_coords | [B,T,5] | Coordenadas 5D |
| phi_total | [B,T,1] | Saude do manifold |
| vi_direction | [B,T,5] | Correcao VI |

## Instalacao

```bash
pip install -e ".[dev,data]"
```

## Testes

```bash
pytest tests/ -v
```

## Treinamento

```bash
# Config pequena (testes rapidos)
python -c "
from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model
from aletheion_v2.training.data import create_synthetic_data
from aletheion_v2.training.trainer import Trainer

config = AletheionV2Config.small()
model = AletheionV2Model(config)
loader = create_synthetic_data(seq_len=config.max_seq_len, batch_size=config.batch_size)
trainer = Trainer(model, config, loader)
trainer.train()
"
```

## Geracao com tomografia

```python
from aletheion_v2 import AletheionV2Config, AletheionV2Model
from aletheion_v2.inference.generator import Generator
from aletheion_v2.inference.dashboard_bridge import DashboardBridge

config = AletheionV2Config.small()
model = AletheionV2Model(config)
gen = Generator(model, max_new_tokens=50, top_k=50)

input_ids = torch.randint(0, config.vocab_size, (1, 10))
result = gen.generate(input_ids)

bridge = DashboardBridge()
snapshot = bridge.from_generation_result(result)
print(bridge.to_json(snapshot))
```

## Loss Composta

```
L = 1.0*CE + 0.1*VARO + 0.01*VI_reg + 0.05*MAD_cal + 0.001*metric_reg
```

Com annealing: warmup 10% so CE, ramp linear ate 50%.

## Licenca

Proprietario. Copyright (c) 2025-2026 Felipe Maya Muniz.
