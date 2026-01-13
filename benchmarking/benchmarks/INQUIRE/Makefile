# INQUIRE Benchmark Makefile
# This file sets INQUIRE-specific variables and includes the base benchmarking Makefile

# ============================================================================
# Required Variables (must be set for base Makefile)
# ============================================================================
BENCHMARK_NAME := inquire
DOCKERFILE_JOB := Dockerfile.job
RESULTS_FILES := image_search_results.csv query_eval_metrics.csv
ENV ?= dev
ifeq ($(ENV),prod)
  KUSTOMIZE_DIR := ../../kubernetes/INQUIRE/nrp-prod
else
  KUSTOMIZE_DIR := ../../kubernetes/INQUIRE/nrp-dev
endif

# ============================================================================
# Optional Variables (can be overridden)
# ============================================================================
KUBECTL_NAMESPACE := sage
KUBECTL_CONTEXT ?= nautilus
REGISTRY := gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search
JOB_TAG ?= latest

# ============================================================================
# INQUIRE-Specific Environment Variables
# ============================================================================
# These are used by the Kubernetes deployments via environment variable patches
INQUIRE_DATASET ?= sagecontinuum/INQUIRE-Benchmark-small
IMAGE_BATCH_SIZE ?= 25
QUERY_BATCH_SIZE ?= 5
SAMPLE_SIZE ?= 0
WORKERS ?= 5
IMAGE_RESULTS_FILE ?= image_search_results.csv
QUERY_EVAL_METRICS_FILE ?= query_eval_metrics.csv

# Local run script
RUN_SCRIPT := run_benchmark.py

# Include the base Makefile (after setting variables)
include ../Makefile
