# Benchmark Dockerfile Guide

## Overview

The benchmarking framework provides a Dockerfile template that can be reused across all benchmarks. Since Dockerfiles must be in the benchmark directory for the build context, each benchmark creates its own Dockerfile based on the template.

## Template

The base template is located at `benchmarks/template/Dockerfile.job`. This template provides:

- Python 3.11 base image (configurable via `PYTHON_VERSION` ARG)
- System dependencies (build-essential, git, procps)
- Requirements installation
- Application code copying
- Entrypoint that runs `run_benchmark.py`

## Creating Dockerfile for a New Benchmark

### Step 1: Copy the Template

Copy the template to your benchmark directory:

```bash
cd benchmarks/MYBENCHMARK
cp ../template/Dockerfile.job Dockerfile.job
```

Or use the complete template directory:

```bash
cd benchmarks
cp -r template MYBENCHMARK
cd MYBENCHMARK
# Customize the files as needed
```

### Step 2: Verify the Entrypoint

The `Dockerfile.job` should run the combined benchmark script:

```dockerfile
# Run combined benchmark script
CMD ["python", "run_benchmark.py"]
```

This is already set in the template, so usually no changes are needed.

### Step 3: Ensure requirements.txt is Complete

Make sure your `requirements.txt` includes:
- `imsearch_eval[weaviate]` - Core benchmarking framework
- `minio>=7.2.0` - S3 upload support
- Any benchmark-specific dependencies

## Dockerfile Structure

A typical `Dockerfile.job` looks like:

```dockerfile
# MYBENCHMARK Benchmark Job Dockerfile
# Combined Dockerfile for running both data loading and evaluation

ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run combined benchmark script
CMD ["python", "run_benchmark.py"]
```

## Building Images

### Local Build

```bash
cd benchmarks/MYBENCHMARK
docker build -f Dockerfile.job -t benchmark-mybenchmark-job:latest .
```

### Using Makefile

```bash
cd benchmarks/MYBENCHMARK
make build
```

This will build the image using the `DOCKERFILE_JOB` specified in the Makefile.

### Using GitHub Actions

The GitHub Actions workflow (`.github/workflows/benchmarking.yml`) automatically builds and pushes images when:
- Changes are pushed to `main` branch
- Tags are created
- Pull requests are opened

The workflow builds `Dockerfile.job` for each benchmark.

## Image Naming Convention

Images should follow this naming pattern:
- `benchmark-{benchmark-name}-job:latest`

For example:
- `benchmark-inquire-job:latest`
- `benchmark-mybenchmark-job:latest`

## Registry

Images are typically pushed to:
- `gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-{name}-job:latest`

Update the `REGISTRY` variable in your Makefile if using a different registry.

## Best Practices

1. **Keep Dockerfiles Simple**: The template is already optimized, avoid unnecessary changes
2. **Pin Dependencies**: Use specific versions in `requirements.txt` when possible
3. **Multi-stage Builds**: Not needed for benchmarks, but can be used if image size is a concern
4. **Layer Caching**: The template is structured to maximize Docker layer caching
5. **Security**: Keep base images updated and avoid running as root if possible

## Troubleshooting

### Build Fails with "Module not found"

Ensure all dependencies are in `requirements.txt` and the file is copied before `pip install`.

### Build is Slow

- Check if Docker layer caching is working
- Consider using a local Docker registry for faster builds
- Ensure `requirements.txt` is copied before application code (for better caching)

### Image is Too Large

- Use `python:3.11-slim` base image (already in template)
- Remove unnecessary system packages after installation
- Consider multi-stage builds if needed

## See Also

- `template/Dockerfile.job` - Complete template example
- `INQUIRE/Dockerfile.job` - Real-world example
- `../README.md` - Framework overview
- `../MAKEFILE.md` - Makefile documentation
