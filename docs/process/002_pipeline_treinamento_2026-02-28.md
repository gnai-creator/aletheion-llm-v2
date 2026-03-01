# 002 - Pipeline de Treinamento Escalavel

**Data:** 2026-02-28
**Autor:** Claude (assistido)

## Resumo

Adicionado pipeline completo para treinamento de 1M a 162B parametros:
- Tokenizer tiktoken com suporte a sentencepiece
- Data pipeline com streaming HuggingFace e sharding memmap
- Treinamento distribuido (DDP/FSDP) com mixed precision
- 11 configs de escala (1M, 10M, 50M, 125M, 350M, 1.3B, 7B, 13B, 30B, 70B, 162B)
- Scripts de preparacao de dados e lancamento de treinamento

## Arquivos Adicionados

### Tokenizer
- `src/aletheion_v2/tokenizer/__init__.py`
- `src/aletheion_v2/tokenizer/tokenizer.py` (195 LOC) - wrapper tiktoken/sentencepiece

### Data Pipeline
- `src/aletheion_v2/training/data_pipeline.py` (260 LOC) - MemmapDataset, ShardedMemmapDataset, StreamingHFDataset, MixedDataset

### Distribuido
- `src/aletheion_v2/training/distributed.py` (195 LOC) - setup DDP/FSDP, mixed precision, gradient checkpointing
- `src/aletheion_v2/training/trainer_distributed.py` (290 LOC) - DistributedTrainer

### Configs de Escala
- `configs/scaling/{1m,10m,50m,125m,350m,1.3b,7b,13b,30b,70b,162b}.yaml`

### Scripts
- `scripts/prepare_data.py` (200 LOC) - download + tokenizacao + sharding
- `scripts/train_distributed.py` (100 LOC) - lancamento de treinamento

## Testes

85 testes existentes continuam passando (nenhuma regressao).
