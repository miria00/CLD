#!/bin/bash

set -euo pipefail

# Configuration
CONFIG_PATH="configs/hi_ms_config.json"
OUT_DIR="data/hi_ms"
COMMON_VOICE_DIR="data/cv/"
MUSAN_DIR="musan/noise/free-sound"

# Ingest data
python data_ingestion.py \
  --config "$CONFIG_PATH" \
  --out "$OUT_DIR" \
  --musan-dir "$MUSAN_DIR" \
  --common-voice-dir "$COMMON_VOICE_DIR"

# Evaluate binary hi vs ms using vanilla language detection (or swap to nn/cvx if trained)
WANDB_NAME="hi-ms-binary-$(date +%Y%m%d-%H%M%S)" python benchmark_cld.py \
  --dataset_path "$OUT_DIR" \
  --whisper_path openai/whisper-small \
  --cld_type vanilla \
  --lang1 hi \
  --lang2 ms \
  --batch_size 8
