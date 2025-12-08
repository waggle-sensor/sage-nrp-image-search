#!/bin/bash
set -euo pipefail

# --- knobs you can tweak via env ---
# NOTE: using 1 worker and disabling HF_HUB_ENABLE_HF_TRANSFER was the fastest for NRP environment
: "${DATA_DIR:=/var/lib/weaviate}"     # where to store the dataset
: "${HF_WORKERS:=1}"                   # parallel download workers 
: "${HF_HOME:=/hf_cache}"              # where to cache the huggingface datasets
: "${HF_HUB_ENABLE_HF_TRANSFER:=0}"    # fast, parallel ranged downloads
: "${PYTHONUNBUFFERED:=1}"             # unbuffer the output
export DATA_DIR HF_WORKERS PYTHONUNBUFFERED HF_HOME HF_HUB_ENABLE_HF_TRANSFER

echo "[init] Initializing environment..."

# Install CLI
pip install --no-cache-dir "huggingface_hub>=0.24" "hf_transfer>=0.1.6"

# Download dataset
mkdir -p "$DATA_DIR"
echo "[init] downloading init dataset (workers=$HF_WORKERS) ..."
hf download \
  "sagecontinuum/init_img_search" \
  --repo-type dataset \
  --local-dir "$DATA_DIR" \
  --cache-dir "$HF_HOME" \
  --max-workers "$HF_WORKERS" \
  --token "$HF_TOKEN"

# Reassemble dataset
echo "[init] rebuilding dataset..."
cd "$DATA_DIR"
chmod +x scripts/reassemble_db.sh
/bin/bash scripts/reassemble_db.sh

# Clean up
echo "[init] cleaning up files used to build the dataset (shards, sha256, part files)..."
rm hybridsearchexample/GwNba1enwnW2/lsm/objects/*.part 
rm hybridsearchexample/GwNba1enwnW2/lsm/objects/*.sha256
rm -rf scripts .cache

# Finalize
echo "[init] $DATA_DIR final size: $(du -sh "$DATA_DIR" 2>/dev/null | awk '{print $1}')"
echo "[init] Done."