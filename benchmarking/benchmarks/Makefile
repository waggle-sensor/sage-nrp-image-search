# Base Makefile for benchmarking framework
# This file contains generic benchmark commands that can be reused by any benchmark instance.
# Each benchmark should define its specific variables and include this file.

.PHONY: down build run kubectl port-forward-start port-forward-stop run-local status logs

# ============================================================================
# Required Variables (must be set by benchmark-specific Makefile)
# ============================================================================
# BENCHMARK_NAME - Name of the benchmark (e.g., "inquire")
# KUSTOMIZE_DIR - Path to kustomize directory (e.g., "../../kubernetes/INQUIRE")
# DOCKERFILE_JOB - Dockerfile for combined job (e.g., "Dockerfile.job")
# RESULTS_FILES - Space-separated list of result files to copy (e.g., "image_search_results.csv query_eval_metrics.csv")

# ============================================================================
# Optional Variables (can be overridden)
# ============================================================================
KUBECTL_NAMESPACE ?= sage
KUBECTL_CONTEXT ?= nautilus
REGISTRY ?= gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search
JOB_TAG ?= latest
ENV ?= dev

# Local run configuration
WEAVIATE_SERVICE ?= $(ENV)-weaviate
TRITON_SERVICE ?= $(ENV)-triton
WEAVIATE_HTTP_PORT ?= 8080
WEAVIATE_GRPC_PORT ?= 50051
TRITON_GRPC_PORT ?= 8001
LOCAL_WEAVIATE_HTTP_PORT ?= 8080
LOCAL_WEAVIATE_GRPC_PORT ?= 50051
LOCAL_TRITON_GRPC_PORT ?= 8001
RUN_SCRIPT ?= run_benchmark.py

# Derived variables
JOB_IMAGE ?= $(REGISTRY)/benchmark-$(BENCHMARK_NAME)-job

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
	@if [ -z "$(DOCKERFILE_JOB)" ]; then \
		echo "Error: DOCKERFILE_JOB must be set"; exit 1; \
	fi
	@echo "Building benchmark job image..."
	@docker build -t $(JOB_IMAGE):$(JOB_TAG) -f $(DOCKERFILE_JOB) .
	@echo "Build complete. Push with:"
	@echo "  docker push $(JOB_IMAGE):$(JOB_TAG)"

# ============================================================================
# Cleanup Targets
# ============================================================================
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

# Run benchmark job (deploys and runs the benchmark)
run: kubectl
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@if [ -z "$(KUSTOMIZE_DIR)" ]; then \
		echo "Error: KUSTOMIZE_DIR must be set"; exit 1; \
	fi
	@echo "Deploying and running $(BENCHMARK_NAME) benchmark job..."
	@kubectl apply -k $(KUSTOMIZE_DIR)
	@echo "Job started. Monitor with:"
	@echo "  kubectl logs -n $(KUBECTL_NAMESPACE) -l app=benchmark-job,benchmark=$(BENCHMARK_NAME) -f"


# ============================================================================
# Utility Targets
# ============================================================================

# Show status
status: kubectl
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@echo "=== $(BENCHMARK_NAME) Benchmark Status ==="
	@echo "=== Jobs ==="
	@kubectl get jobs -n $(KUBECTL_NAMESPACE) -l benchmark=$(BENCHMARK_NAME)
	@echo ""
	@echo "=== Pods ==="
	@kubectl get pods -n $(KUBECTL_NAMESPACE) -l benchmark=$(BENCHMARK_NAME)

# Show logs
logs: kubectl
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@pod=$$(kubectl get pods -n $(KUBECTL_NAMESPACE) -l app=benchmark-job,benchmark=$(BENCHMARK_NAME) --field-selector=status.phase!=Failed,status.phase!=Pending --sort-by=.metadata.creationTimestamp -o jsonpath="{.items[-1].metadata.name}"); \
	if [ -z "$$pod" ]; then \
		echo "No Running or Completed pods found for $(BENCHMARK_NAME) benchmark."; \
		exit 1; \
	fi; \
	phase=$$(kubectl get pod -n $(KUBECTL_NAMESPACE) $$pod -o jsonpath="{.status.phase}"); \
	if [ "$$phase" = "Running" ]; then \
		kubectl logs -n $(KUBECTL_NAMESPACE) -f $$pod; \
	else \
		kubectl logs -n $(KUBECTL_NAMESPACE) $$pod; \
	fi

# ============================================================================
# Local Development Targets
# ============================================================================

# Start port-forwarding for Weaviate and Triton services
port-forward-start: kubectl
	@echo "Starting port-forwarding for services..."
	@echo "Weaviate HTTP: localhost:$(LOCAL_WEAVIATE_HTTP_PORT) -> $(WEAVIATE_SERVICE):$(WEAVIATE_HTTP_PORT)"
	@echo "Weaviate gRPC: localhost:$(LOCAL_WEAVIATE_GRPC_PORT) -> $(WEAVIATE_SERVICE):$(WEAVIATE_GRPC_PORT)"
	@echo "Triton gRPC: localhost:$(LOCAL_TRITON_GRPC_PORT) -> $(TRITON_SERVICE):$(TRITON_GRPC_PORT)"
	@kubectl port-forward -n $(KUBECTL_NAMESPACE) svc/$(WEAVIATE_SERVICE) $(LOCAL_WEAVIATE_HTTP_PORT):$(WEAVIATE_HTTP_PORT) $(LOCAL_WEAVIATE_GRPC_PORT):$(WEAVIATE_GRPC_PORT) > /tmp/kubectl-port-forward-weaviate.log 2>&1 & \
		echo $$! > /tmp/kubectl-port-forward-weaviate.pid
	@kubectl port-forward -n $(KUBECTL_NAMESPACE) svc/$(TRITON_SERVICE) $(LOCAL_TRITON_GRPC_PORT):$(TRITON_GRPC_PORT) > /tmp/kubectl-port-forward-triton.log 2>&1 & \
		echo $$! > /tmp/kubectl-port-forward-triton.pid
	@sleep 2
	@echo "Port-forwarding started. PIDs saved to /tmp/kubectl-port-forward-*.pid"
	@echo "To stop port-forwarding, run: make port-forward-stop"

# Stop port-forwarding
port-forward-stop:
	@echo "Stopping port-forwarding..."
	@if [ -f /tmp/kubectl-port-forward-weaviate.pid ]; then \
		kill $$(cat /tmp/kubectl-port-forward-weaviate.pid) 2>/dev/null || true; \
		rm -f /tmp/kubectl-port-forward-weaviate.pid; \
		echo "Stopped Weaviate port-forwarding"; \
	fi
	@if [ -f /tmp/kubectl-port-forward-triton.pid ]; then \
		kill $$(cat /tmp/kubectl-port-forward-triton.pid) 2>/dev/null || true; \
		rm -f /tmp/kubectl-port-forward-triton.pid; \
		echo "Stopped Triton port-forwarding"; \
	fi
	@echo "Port-forwarding stopped."

# Run benchmark locally with port-forwarding
run-local: port-forward-start
	@if [ -z "$(BENCHMARK_NAME)" ]; then \
		echo "Error: BENCHMARK_NAME must be set"; exit 1; \
	fi
	@if [ ! -f "$(RUN_SCRIPT)" ]; then \
		echo "Error: Run script $(RUN_SCRIPT) not found"; exit 1; \
	fi
	@echo "Running $(BENCHMARK_NAME) benchmark locally..."
	@echo "Using port-forwarded services:"
	@echo "  WEAVIATE_HOST=127.0.0.1"
	@echo "  WEAVIATE_PORT=$(LOCAL_WEAVIATE_HTTP_PORT)"
	@echo "  WEAVIATE_GRPC_PORT=$(LOCAL_WEAVIATE_GRPC_PORT)"
	@echo "  TRITON_HOST=127.0.0.1"
	@echo "  TRITON_PORT=$(LOCAL_TRITON_GRPC_PORT)"
	@WEAVIATE_HOST=127.0.0.1 \
	 WEAVIATE_PORT=$(LOCAL_WEAVIATE_HTTP_PORT) \
	 WEAVIATE_GRPC_PORT=$(LOCAL_WEAVIATE_GRPC_PORT) \
	 TRITON_HOST=127.0.0.1 \
	 TRITON_PORT=$(LOCAL_TRITON_GRPC_PORT) \
	 python $(RUN_SCRIPT) || (make port-forward-stop && exit 1)
	@make port-forward-stop
	@echo "Benchmark run completed. Results saved locally."
