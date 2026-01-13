# Sage NRP Image Search - Kubernetes Deployment

This folder contains the Kubernetes manifests for deploying the `sage-nrp-image-search` stack on Nautilus or other Kubernetes clusters. It provides all the core resources and configuration required for running the hybrid image search service, but **does not** include benchmark configs or benchmark jobs.

## Contents

- `base/`: Base kustomize configuration and manifests for core deployment
- `base/kustomization.yaml`: Main kustomization file listing services, secrets, and configMaps
- `base/*.yaml`: Service, Deployment, Job, and Secret manifests for all core components (Weaviate, Triton, Reranker, Gradio UI, etc.)

## Deployment Overview

The resources here stand up the core application stack:

- **Weaviate** (vector database)
- **Triton** (inference server)
- **Reranker Transformers** (optional re-ranking model)
- **Gradio UI**
- **Support jobs** for dataset management, storage, and configuration
- **Secrets** for Hugging Face, S3, and Sage user credentials

All roles and deployments are configured using kustomize to simplify environment management and overlays.

## Setting Up Secrets

Before deploying, you must create the necessary secret manifest files in `base/`. Templates are provided for all required secrets:

### 1. HuggingFace Secret

Copy the template and fill in your HuggingFace token (base64-encoded):

```bash
cp base/huggingface-secret.template.yaml base/._huggingface-secret.yaml
```

### 2. Sage User Secret

Copy the Sage user secret template and add your Sage account name and password:

```bash
cp base/sage-user-secret.template.yaml base/._sage-user-secret.yaml
```

- Encode username and password values as above.
- Update the `SAGE_USER` and `SAGE_PASS` fields.

> **Important:** 
> All secret files you actually use must be named with leading `._` per `.gitignore` and not checked into version control! Only commit the `*.template.yaml` files.

## Deploying

> Prerequisites:
> - `kubectl` configured with cluster access
> - `kustomize`

To deploy the base stack:

```bash
cd kubernetes/base
kustomize build . | kubectl apply -f -
```

Or, using kubectl (if it supports native kustomize):

```bash
kubectl apply -k base/
```

## Managing and Customizing

You can extend or patch this `base/` deployment using kustomize overlays for different environments, resource limits, or development setups. See included overlays (such as those in benchmark subfolders) for example usage.

## Note

- These resources do **not** include benchmark job definitions. For benchmarking, see `benchmarking/kubernetes/`.
- Update secret files as needed to match your deployment’s authentication requirements.