# Sage NRP Image Search - Kubernetes Deployment

This folder contains the Kubernetes manifests for deploying the `sage-nrp-image-search` stack on Nautilus or other Kubernetes clusters. It provides all the core resources and configuration required for running the hybrid image search service, but **does not** include benchmark configs or benchmark jobs.

## Contents

- `base/`: Base kustomize configuration and manifests for core deployment
- `base/kustomization.yaml`: Main kustomization file listing services, secrets, and configMaps
- `base/*.yaml`: Service, Deployment, Job, and Secret manifests for core components (Triton, Gradio UI, weavloader, weavmanage, etc.)

## Deployment Overview

The resources here stand up the core application stack:

- **NRP-managed Milvus** (vector database — no Milvus pod; URI/token via `milvus-secret`)
- **Triton** (inference server)
- **Gradio UI** (hybrid search + Triton CLIP rerank)
- **Weavloader / Weavmanage** (ingest + schema Job; internals use pymilvus)
- **Secrets** for Hugging Face, Sage user, NRP LLM, and Milvus credentials

All roles and deployments are configured using kustomize to simplify environment management and overlays.

Collection names per env:

| Overlay | `MILVUS_COLLECTION` |
|---------|---------------------|
| nrp-prod | `HybridSearchExample` |
| nrp-dev / prs | `HybridSearchExampleDev` |

## Setting Up Secrets

Before deploying, you must create the necessary secret manifest files in `base/`. Templates are provided for all required secrets:

### 1. HuggingFace Secret

Copy the template and fill in your HuggingFace token (base64-encoded):

```bash
cp base/huggingface-secret.template.yaml base/._huggingface-secret.yaml
```

To generate base64 encoded Hugging Face token:
```
echo -n "your_hf_token_here" | base64
```

### 2. Sage User Secret

Copy the Sage user secret template and add your Sage account name and password:

```bash
cp base/sage-user-secret.template.yaml base/._sage-user-secret.yaml
```

Base64 encoded SAGE_USER and SAGE_PASS to generate:
```
echo -n "your_username_here" | base64
echo -n "your_password_here" | base64
```

- Update the `SAGE_USER` and `SAGE_PASS` fields.

> **Important:** 
> All secret files you actually use must be named with leading `._` per `.gitignore` and not checked into version control! Only commit the `*.template.yaml` files.

### 3. NRP LLMs (optional)

Copy the NRP LLM user secret template and add your NRP LLM API endpoint and token:
```bash
cp base/nrp-llm-user-secret.template.yaml base/._nrp-llm-user-secret.yaml
```

Base64 encoded NRP_API_ENDPOINT and NRP_API_KEY to generate:
```
echo -n "your_username_here" | base64
echo -n "your_password_here" | base64
```

### 4. Milvus Secret

```bash
cp base/milvus-secret.template.yaml base/._milvus-secret.yaml
```

Fill base64-encoded `MILVUS_URI` (default `https://milvus.nrp-nautilus.io:50051`) and `MILVUS_TOKEN` (`user:password`) from the [NRP vector-database docs](https://nrp.ai/documentation/userdocs/ai/vector-database/):

```
echo -n "https://milvus.nrp-nautilus.io:50051" | base64
echo -n "username:password" | base64
```

Collections are created in `MILVUS_DB=image_search_svc` (database already provisioned by NRP admins).


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

Deploy all services:
```
kubectl kustomize nrp-dev | kubectl apply -f -
kubectl kustomize nrp-prod | kubectl apply -f -
```

Delete all services:
```
kubectl kustomize nrp-dev | kubectl delete -f -
kubectl kustomize nrp-prod | kubectl delete -f -
```

Debugging - output to yaml:
```
kubectl kustomize nrp-dev -o hybrid-search-dev.yaml
kubectl kustomize nrp-prod -o hybrid-search-dev.yaml
```

## Testing a Pull Request
For testing a Pull Request (PR), the overlay [prs](/kubernetes/prs/) is provided. Github Actions is setup to create an image for each PR so that we can manually test or in the future automatically test an instance of the image search deployed on k8s.

The following manual steps are required for now:
- [kubernetes/prs/kustomization.yaml](/kubernetes/prs/kustomization.yaml)
    - change the `namePrefix` to the name of the PR
    - change `commonLabels.env` to the name of the PR
    - change the `newTag` to the name of the PR for each service that needs it
- port-forwarding for any of the services to test out (update `pr`):
    - `kubectl port-forward svc/pr-triton 8001:8001`: triton endpoint to call the LLM models locally
    - `kubectl port-forward svc/pr-gradio-ui 7860:7860`: Search UI
    - `kubectl port-forward svc/pr-weavloader-metrics 5555:5555`: Weavloader Flower endpoint
    - `kubectl port-forward svc/pr-weavloader-metrics 8081:8080`: Weavloader Prometheus endpoint

Deploy:
```
kubectl kustomize prs | kubectl apply -f -
```

Delete all services:
```
kubectl kustomize prs | kubectl delete -f -
```

Debugging - output to yaml:
```
kubectl kustomize prs -o hybrid-search-pr.yaml
```

Notes:
- Make sure that your PR is up-to-date with `main` so that the services that were not modified are reflected for the `latest` tag. This can be also be checked with the [docker-compose](/docker-compose.yml) local deployment (after the PR is up-to-date with `main`) to see if the changes in the PR are working with the rest of the services that were not modified.
- Users can utilized this overlay to combine it with their local docker compose instance to use a triton instance that has an NVIDIA GPU. This involves commenting out the ports from the docker compose manifest file for triton and doing the kubectl port-forwarding described above.

## Managing and Customizing

You can extend or patch this `base/` deployment using kustomize overlays for different environments, resource limits, or development setups. See included overlays (such as those in benchmark subfolders) for example usage.

## Note

After cutover, tear down legacy Weaviate / reranker Deployments and PVCs once the UI smoke-tests cleanly against Milvus.
