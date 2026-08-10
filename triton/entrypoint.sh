#!/bin/bash

set -e

CLIP_HF_REPO="${CLIP_HF_REPO:-apple/DFN5B-CLIP-ViT-H-14-378}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/models/clip}"
CLIP_MODEL_VERSION="${CLIP_MODEL_VERSION:-419d1f8f6a96aabaf5913c526d059facda50c24b}"

GEMMA_HF_REPO="${GEMMA_HF_REPO:-google/gemma-3-4b-it}"
GEMMA_MODEL_PATH="${GEMMA_MODEL_PATH:-/models/gemma-3-4b-it}"
GEMMA_MODEL_VERSION="${GEMMA_MODEL_VERSION:-093f9f388b31de276ce2de164bdc2081324b9767}"

clip_model_ready() {
  local dir="$1"
  [[ -f "$dir/config.json" ]] \
    && [[ -f "$dir/preprocessor_config.json" ]] \
    && { [[ -f "$dir/model.safetensors" ]] || compgen -G "$dir/"*.safetensors >/dev/null; } \
    && [[ -f "$dir/tokenizer_config.json" || -f "$dir/tokenizer.json" ]]
}

# Gemma 3 multimodal needs processor + image preprocessor + tokenizer + weights.
# A non-empty directory is not enough: a partial HF download leaves Triton failing
# AutoProcessor with "Unrecognized processing class".
gemma_model_ready() {
  local dir="$1"
  [[ -f "$dir/config.json" ]] \
    && grep -q '"model_type"[[:space:]]*:[[:space:]]*"gemma3"' "$dir/config.json" \
    && [[ -f "$dir/processor_config.json" ]] \
    && [[ -f "$dir/preprocessor_config.json" ]] \
    && [[ -f "$dir/tokenizer_config.json" ]] \
    && [[ -f "$dir/tokenizer.json" || -f "$dir/tokenizer.model" ]] \
    && { [[ -f "$dir/model.safetensors" ]] || [[ -f "$dir/model.safetensors.index.json" ]] \
         || compgen -G "$dir/model-*.safetensors" >/dev/null; }
}

download_clip() {
  echo "Downloading CLIP model from ${CLIP_HF_REPO} (revision: ${CLIP_MODEL_VERSION}) to ${CLIP_MODEL_PATH}..."
  mkdir -p "$CLIP_MODEL_PATH"
  # CLIP is public; clear token so gated-auth quirks cannot interfere.
  HF_TOKEN= huggingface-cli download \
      --local-dir "$CLIP_MODEL_PATH" \
      --revision "$CLIP_MODEL_VERSION" \
      "$CLIP_HF_REPO"
}

download_gemma() {
  echo "Downloading Gemma model from ${GEMMA_HF_REPO} (revision: ${GEMMA_MODEL_VERSION}) to ${GEMMA_MODEL_PATH}..."
  mkdir -p "$GEMMA_MODEL_PATH"
  huggingface-cli download \
      --local-dir "$GEMMA_MODEL_PATH" \
      --revision "$GEMMA_MODEL_VERSION" \
      "$GEMMA_HF_REPO"
}

# --- CLIP ---
if clip_model_ready "$CLIP_MODEL_PATH"; then
  echo "CLIP model already present at ${CLIP_MODEL_PATH}. Skipping download."
else
  if [[ -d "$CLIP_MODEL_PATH" ]] && [[ -n "$(ls -A "$CLIP_MODEL_PATH" 2>/dev/null)" ]]; then
    echo "CLIP model at ${CLIP_MODEL_PATH} is incomplete; re-downloading..."
    rm -rf "${CLIP_MODEL_PATH:?}/"*
  fi
  download_clip
  if ! clip_model_ready "$CLIP_MODEL_PATH"; then
    echo "ERROR: CLIP download finished but required files are still missing in ${CLIP_MODEL_PATH}" >&2
    ls -la "$CLIP_MODEL_PATH" >&2 || true
    exit 1
  fi
fi

# --- Gemma (gated; requires HF_TOKEN with license accepted) ---
if [[ -n "$HF_TOKEN" ]]; then
  if gemma_model_ready "$GEMMA_MODEL_PATH"; then
    echo "Gemma model already present at ${GEMMA_MODEL_PATH}. Skipping download."
  else
    if [[ -d "$GEMMA_MODEL_PATH" ]] && [[ -n "$(ls -A "$GEMMA_MODEL_PATH" 2>/dev/null)" ]]; then
      echo "Gemma model at ${GEMMA_MODEL_PATH} is incomplete; re-downloading..."
      echo "Present files:"
      ls -la "$GEMMA_MODEL_PATH" || true
      rm -rf "${GEMMA_MODEL_PATH:?}/"*
    fi
    download_gemma
    if ! gemma_model_ready "$GEMMA_MODEL_PATH"; then
      echo "ERROR: Gemma download finished but required processor/model files are missing in ${GEMMA_MODEL_PATH}" >&2
      echo "Ensure HF_TOKEN can access ${GEMMA_HF_REPO} (accept the model license on Hugging Face)." >&2
      ls -la "$GEMMA_MODEL_PATH" >&2 || true
      exit 1
    fi
  fi
else
  echo "HF_TOKEN not provided. Skipping Hugging Face Gemma download."
fi

# Keep serving healthy models (e.g. CLIP) even if optional ones fail to load.
exec tritonserver \
  --model-repository="$MODEL_REPOSITORY" \
  --exit-on-error=false \
  --strict-readiness=false \
  "$@"
