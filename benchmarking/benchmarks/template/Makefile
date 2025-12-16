# MYBENCHMARK Benchmark Makefile
# This file sets MYBENCHMARK-specific variables and includes the base benchmarking Makefile
#
# Instructions:
# 1. Replace MYBENCHMARK with your benchmark name throughout this file
# 2. Update the required variables below
# 3. Add any benchmark-specific environment variables at the bottom

# Include the base Makefile (after setting variables)
include ../Makefile

# ============================================================================
# Required Variables (must be set for base Makefile)
# ============================================================================
# TODO: Replace MYBENCHMARK with your benchmark name
BENCHMARK_NAME := mybenchmark
KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK
DOCKERFILE_EVALUATOR := Dockerfile.benchmark
DOCKERFILE_DATA_LOADER := Dockerfile.data_loader
RESULTS_PVC_NAME := $(BENCHMARK_NAME)-benchmark-results-pvc
RESULTS_FILES := results.csv metrics.csv

# ============================================================================
# Optional Variables (can be overridden)
# ============================================================================
KUBECTL_NAMESPACE := sage
KUBECTL_CONTEXT ?= nrp-dev
REGISTRY := gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search
EVALUATOR_TAG ?= latest
DATA_LOADER_TAG ?= latest

# ============================================================================
# MYBENCHMARK-Specific Environment Variables
# ============================================================================
# These are used by the Kubernetes deployments via environment variable patches
# TODO: Add your benchmark-specific environment variables here
# Example:
# MYBENCHMARK_DATASET ?= mydataset/benchmark
# BATCH_SIZE ?= 10
# WORKERS ?= 5

