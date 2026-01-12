# MYBENCHMARK Benchmark Makefile
# This file sets MYBENCHMARK-specific variables and includes the base benchmarking Makefile
#
# Instructions:
# 1. Replace MYBENCHMARK with your benchmark name throughout this file
# 2. Update the required variables below
# 3. Add any benchmark-specific environment variables at the bottom

# ============================================================================
# Required Variables (must be set for base Makefile)
# ============================================================================
# TODO: Replace MYBENCHMARK with your benchmark name
BENCHMARK_NAME := mybenchmark
DOCKERFILE_JOB := Dockerfile.job
RESULTS_FILES := image_search_results.csv query_eval_metrics.csv
ENV ?= dev
ifeq ($(ENV),prod)
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK/nrp-prod
else
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK
endif

# ============================================================================
# Optional Variables (can be overridden)
# ============================================================================
KUBECTL_NAMESPACE := sage
KUBECTL_CONTEXT ?= nautilus
REGISTRY := gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search
JOB_TAG ?= latest

# ============================================================================
# MYBENCHMARK-Specific Environment Variables
# ============================================================================
# These are used by the Kubernetes deployments via environment variable patches
# TODO: Add your benchmark-specific environment variables here
# Example:
# MYBENCHMARK_DATASET ?= mydataset/benchmark
# BATCH_SIZE ?= 10
# WORKERS ?= 5

# Local run script
RUN_SCRIPT := run_benchmark.py

# Include the base Makefile (after setting variables)
include ../Makefile
