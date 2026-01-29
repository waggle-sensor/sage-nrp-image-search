#!/bin/bash

set -e

#Download CLIP model if not already present and check if directory is empty
if [ ! -d "$CLIP_MODEL_PATH" ] || [ -z "$(ls -A "$CLIP_MODEL_PATH" 2>/dev/null)" ]; then
  echo "Downloading CLIP model..."
  HF_TOKEN= huggingface-cli download \
      --local-dir "$CLIP_MODEL_PATH" \
      --revision "$CLIP_MODEL_VERSION" \
      apple/DFN5B-CLIP-ViT-H-14-378
else
  echo "CLIP model already present. Skipping download."
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
