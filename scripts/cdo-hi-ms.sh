#!/bin/bash

set -euo pipefail

# Configuration
CONFIG_PATH="configs/cdo_hi_ms_config.json"
OUT_DIR="data/cdo_hi_ms"
COMMON_VOICE_DIR="../CLD/data/cv-corpus-22.0-2025-06-20/"
MUSAN_DIR="musan/noise/free-sound"

# Ingest data
python data_ingestion.py \
  --config "$CONFIG_PATH" \
  --out "$OUT_DIR" \
  --musan-dir "$MUSAN_DIR" \
  --common-voice-dir "$COMMON_VOICE_DIR"

# Train with CRONOS trainer on ingested dataset
RUN_NAME="cdo-hi-ms-$(date +%Y%m%d-%H%M%S)"
python cronos_trainer.py \
  --model_name "$RUN_NAME" \
  --data_dir "$OUT_DIR" \
  --target_lang "cdo" \
  --output_dir "outputs/$RUN_NAME"
