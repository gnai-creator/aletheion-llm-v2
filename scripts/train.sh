#!/bin/bash
# Treinamento do AletheionV2
# Uso: bash scripts/train.sh [small|medium]

CONFIG=${1:-small}
python -m aletheion_v2.training.trainer --config configs/${CONFIG}.yaml
