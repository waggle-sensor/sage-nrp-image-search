# Base Makefile for benchmarking framework
# This file contains generic benchmark commands that can be reused by any benchmark instance.
# Each benchmark should define its specific variables and include this file.

.PHONY: down build deploy load calculate get get-pvc clean status logs-evaluator logs-data-loader kubectl

# ============================================================================
# Required Variables (must be set by benchmark-specific Makefile)
# ============================================================================
# BENCHMARK_NAME - Name of the benchmark (e.g., "inquire")
# KUSTOMIZE_DIR - Path to kustomize directory (e.g., "../../kubernetes/INQUIRE")
# DOCKERFILE_EVALUATOR - Dockerfile for evaluator (e.g., "Dockerfile.benchmark")
# DOCKERFILE_DATA_LOADER - Dockerfile for data loader (e.g., "Dockerfile.data_loader")
# RESULTS_PVC_NAME - Name of the results PVC (e.g., "$(BENCHMARK_NAME)-benchmark-results-pvc")
# RESULTS_FILES - Space-separated list of result files to copy (e.g., "image_search_results.csv query_eval_metrics.csv")

# ============================================================================
# Optional Variables (can be overridden)
# ============================================================================
KUBECTL_NAMESPACE ?= sage
KUBECTL_CONTEXT ?= nautilus
REGISTRY ?= gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search
EVALUATOR_TAG ?= latest
DATA_LOADER_TAG ?= latest

# Derived variables
EVALUATOR_IMAGE ?= $(REGISTRY)/benchmark-$(BENCHMARK_NAME)-evaluator
DATA_LOADER_IMAGE ?= $(REGISTRY)/benchmark-$(BENCHMARK_NAME)-data-loader
DEPLOYMENT_PREFIX ?= $(BENCHMARK_NAME)-benchmark

# ============================================================================
# Helper Targets
# ============================================================================

# Set kubectl context
kubectl:
	@kubectl config use-context $(KUBECTL_CONTEXT)

# ============================================================================
# Build Targets
# ============================================================================

# Build Docker images (for pushing to registry)
build:
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@if [ -z "$(DOCKERFILE_EVALUATOR)" ]; then \
		echo "Error: DOCKERFILE_EVALUATOR must be set"; exit 1; \
	fi
	@if [ -z "$(DOCKERFILE_DATA_LOADER)" ]; then \
		echo "Error: DOCKERFILE_DATA_LOADER must be set"; exit 1; \
	fi
	@echo "Building benchmark evaluator image..."
	@docker build -t $(EVALUATOR_IMAGE):$(EVALUATOR_TAG) -f $(DOCKERFILE_EVALUATOR) .
	@echo "Building data loader image..."
	@docker build -t $(DATA_LOADER_IMAGE):$(DATA_LOADER_TAG) -f $(DOCKERFILE_DATA_LOADER) .
	@echo "Build complete. Push with:"
	@echo "  docker push $(EVALUATOR_IMAGE):$(EVALUATOR_TAG)"
	@echo "  docker push $(DATA_LOADER_IMAGE):$(DATA_LOADER_TAG)"

# ============================================================================
# Deployment Targets
# ============================================================================

# Deploy to Kubernetes using kustomize
deploy: kubectl
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@if [ -z "$(KUSTOMIZE_DIR)" ]; then \
		echo "Error: KUSTOMIZE_DIR must be set"; exit 1; \
	fi
	@echo "Deploying $(BENCHMARK_NAME) benchmark to Kubernetes..."
	@kubectl apply -k $(KUSTOMIZE_DIR)
	@echo "Deployment complete. Check status with:"
	@echo "  kubectl get pods -n $(KUBECTL_NAMESPACE) -l benchmark=$(BENCHMARK_NAME)"

# Clean up deployments
down: kubectl
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@if [ -z "$(KUSTOMIZE_DIR)" ]; then \
		echo "Error: KUSTOMIZE_DIR must be set"; exit 1; \
	fi
	@echo "Removing $(BENCHMARK_NAME) benchmark deployments..."
	@kubectl delete -k $(KUSTOMIZE_DIR) --ignore-not-found=true || true
	@echo "Cleanup complete."

# ============================================================================
# Execution Targets
# ============================================================================

# Load data into vector database
load: kubectl
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@echo "Scaling down evaluator to free resources..."
	@kubectl scale deployment/$(DEPLOYMENT_PREFIX)-evaluator --replicas=0 -n $(KUBECTL_NAMESPACE) --ignore-not-found=true || true
	@echo "Starting data loader job..."
	@kubectl scale deployment/$(DEPLOYMENT_PREFIX)-data-loader --replicas=1 -n $(KUBECTL_NAMESPACE) --ignore-not-found=true || true
	@echo "Data loader started. Monitor with:"
	@echo "  kubectl logs -n $(KUBECTL_NAMESPACE) -l app=benchmark-data-loader,benchmark=$(BENCHMARK_NAME) -f"

# Run benchmark evaluation
calculate: kubectl
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@echo "Scaling down data loader to free resources..."
	@kubectl scale deployment/$(DEPLOYMENT_PREFIX)-data-loader --replicas=0 -n $(KUBECTL_NAMESPACE) --ignore-not-found=true || true
	@echo "Starting benchmark evaluation..."
	@kubectl scale deployment/$(DEPLOYMENT_PREFIX)-evaluator --replicas=1 -n $(KUBECTL_NAMESPACE) --ignore-not-found=true || true
	@echo "Benchmark evaluator started. Monitor with:"
	@echo "  kubectl logs -n $(KUBECTL_NAMESPACE) -l app=benchmark-evaluator,benchmark=$(BENCHMARK_NAME) -f"

# ============================================================================
# Results Targets
# ============================================================================

# Get results from pod
get: kubectl
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@if [ -z "$(RESULTS_FILES)" ]; then \
		echo "Warning: RESULTS_FILES not set, using default result files"; \
		RESULTS_FILES="image_search_results.csv query_eval_metrics.csv"; \
	fi
	@echo "Retrieving results..."
	@POD_NAME=$$(kubectl get pods -n $(KUBECTL_NAMESPACE) -l app=benchmark-evaluator,benchmark=$(BENCHMARK_NAME) -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) && \
	if [ -n "$$POD_NAME" ]; then \
		echo "Copying results from pod $$POD_NAME..."; \
		for file in $(RESULTS_FILES); do \
			kubectl cp $(KUBECTL_NAMESPACE)/$$POD_NAME:/app/results/$$file ./$$file 2>/dev/null || \
			kubectl cp $(KUBECTL_NAMESPACE)/$$POD_NAME:/app/$$file ./$$file 2>/dev/null || \
			echo "Could not copy $$file"; \
		done; \
		echo "Results retrieved."; \
	else \
		if [ -n "$(RESULTS_PVC_NAME)" ]; then \
			echo "No evaluator pod found. Results may be in PVC: $(RESULTS_PVC_NAME)"; \
		else \
			echo "No evaluator pod found."; \
		fi; \
	fi

# Get results from PVC (alternative method)
get-pvc: kubectl
	@if [ -z "$(RESULTS_PVC_NAME)" ]; then \
		echo "Error: RESULTS_PVC_NAME must be set"; exit 1; \
	fi
	@echo "Creating temporary pod to access results PVC..."
	@kubectl run -it --rm --restart=Never results-copy-$$(date +%s) \
		-n $(KUBECTL_NAMESPACE) \
		--image=busybox \
		--overrides='{"spec":{"volumes":[{"name":"results","persistentVolumeClaim":{"claimName":"$(RESULTS_PVC_NAME)"}}],"containers":[{"name":"results-copy","image":"busybox","volumeMounts":[{"name":"results","mountPath":"/results"}],"command":["sh","-c","ls -la /results && sleep 3600"]}]}}' || true

# ============================================================================
# Utility Targets
# ============================================================================

# Clean up everything
clean: down
	@if [ -n "$(RESULTS_PVC_NAME)" ]; then \
		echo "Cleaning up PVCs..."; \
		kubectl delete pvc $(RESULTS_PVC_NAME) -n $(KUBECTL_NAMESPACE) --ignore-not-found=true || true; \
	fi
	@echo "Cleanup complete."

# Show status
status: kubectl
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@echo "=== $(BENCHMARK_NAME) Benchmark Status ==="
	@kubectl get pods -n $(KUBECTL_NAMESPACE) -l benchmark=$(BENCHMARK_NAME)
	@echo ""
	@echo "=== Services ==="
	@kubectl get svc -n $(KUBECTL_NAMESPACE) -l benchmark=$(BENCHMARK_NAME)
	@echo ""
	@echo "=== PVCs ==="
	@kubectl get pvc -n $(KUBECTL_NAMESPACE) -l benchmark=$(BENCHMARK_NAME)

# Show logs
logs-evaluator: kubectl
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@kubectl logs -n $(KUBECTL_NAMESPACE) -l app=benchmark-evaluator,benchmark=$(BENCHMARK_NAME) -f

logs-data-loader: kubectl
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@kubectl logs -n $(KUBECTL_NAMESPACE) -l app=benchmark-data-loader,benchmark=$(BENCHMARK_NAME) -f

