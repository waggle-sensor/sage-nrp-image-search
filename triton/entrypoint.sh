#!/bin/bash

set -e

CLIP_HF_REPO="${CLIP_HF_REPO:-apple/DFN5B-CLIP-ViT-H-14-378}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/models/clip}"
CLIP_MODEL_VERSION="${CLIP_MODEL_VERSION:-419d1f8f6a96aabaf5913c526d059facda50c24b}"

# Download CLIP model if not already present and check if directory is empty
if [ ! -d "$CLIP_MODEL_PATH" ] || [ -z "$(ls -A "$CLIP_MODEL_PATH" 2>/dev/null)" ]; then
  echo "Downloading CLIP model from ${CLIP_HF_REPO} (revision: ${CLIP_MODEL_VERSION}) to ${CLIP_MODEL_PATH}..."
  HF_TOKEN= huggingface-cli download \
      --local-dir "$CLIP_MODEL_PATH" \
      --revision "$CLIP_MODEL_VERSION" \
      "$CLIP_HF_REPO"
else
  echo "CLIP model already present at ${CLIP_MODEL_PATH}. Skipping download."
fi

# Log in and download models if HF_TOKEN provided
if [ -n "$HF_TOKEN" ]; then
  # Download Gemma model if not already present and check if directory is empty
  if [ ! -d "$GEMMA_MODEL_PATH" ] || [ -z "$(ls -A "$GEMMA_MODEL_PATH" 2>/dev/null)" ]; then
    echo "Downloading Gemma model..."
    # export HF_TOKEN="$HF_TOKEN"
    huggingface-cli download \
      --local-dir "$GEMMA_MODEL_PATH" \
      --revision "$GEMMA_MODEL_VERSION" \
      google/gemma-3-4b-it
else
    echo "Gemma model already present. Skipping download."
  fi
else
  echo "HF_TOKEN not provided. Skipping Hugging Face model downloads."
fi

# Start Triton Inference Server
exec tritonserver --model-repository=$MODEL_REPOSITORY "$@"
