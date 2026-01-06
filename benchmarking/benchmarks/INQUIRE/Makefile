# INQUIRE Benchmark Makefile
# This file sets INQUIRE-specific variables and includes the base benchmarking Makefile

# ============================================================================
# Required Variables (must be set for base Makefile)
# ============================================================================
BENCHMARK_NAME := inquire
KUSTOMIZE_DIR := ../../kubernetes/INQUIRE
DOCKERFILE_EVALUATOR := Dockerfile.benchmark
DOCKERFILE_DATA_LOADER := Dockerfile.data_loader
RESULTS_PVC_NAME := $(BENCHMARK_NAME)-benchmark-results-pvc
RESULTS_FILES := image_search_results.csv query_eval_metrics.csv

# ============================================================================
# Optional Variables (can be overridden)
# ============================================================================
KUBECTL_NAMESPACE := sage
KUBECTL_CONTEXT ?= nautilus
REGISTRY := gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search
EVALUATOR_TAG ?= latest
DATA_LOADER_TAG ?= latest

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

# Include the base Makefile (after setting variables)
include ../Makefile
