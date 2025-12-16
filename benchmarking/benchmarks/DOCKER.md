# Benchmark Dockerfile Guide

## Overview

The benchmarking framework provides a Dockerfile template that can be reused across all benchmarks. Since Dockerfiles must be in the benchmark directory for the build context, each benchmark creates its own Dockerfiles based on the template.

## Template

The base template is located at `benchmarks/template/Dockerfile.template` or `benchmarks/Dockerfile.template`. This template provides:

- Python 3.11 base image (configurable via `PYTHON_VERSION` ARG)
- System dependencies (build-essential)
- Requirements installation
- Application code copying
- Python path configuration for framework, adapters, and app directories
- Configurable entrypoint script

## Creating Dockerfiles for a New Benchmark

### Step 1: Copy the Template

Copy the template to your benchmark directory:

```bash
cd benchmarks/MYBENCHMARK
cp ../template/Dockerfile.template Dockerfile.benchmark
cp ../template/Dockerfile.template Dockerfile.data_loader
# Or use the template in benchmarks root:
# cp ../Dockerfile.template Dockerfile.benchmark
```

Or use the complete template directory:

```bash
cd benchmarks
cp -r template MYBENCHMARK
cd MYBENCHMARK
# Customize the files as needed
```

### Step 2: Customize the Entrypoint

**For `Dockerfile.benchmark` (evaluator):**
```dockerfile
# Run benchmark evaluator
CMD ["python", "main.py"]
```

**For `Dockerfile.data_loader` (data loader):**
```dockerfile
# Run data loader
CMD ["python", "load_data.py"]
```

### Step 3: Install Dependencies

The `imsearch-eval` package is installed via `requirements.txt`. No need to set `PYTHONPATH` as the package is installed in the Python environment.

Make sure your `requirements.txt` includes:
```txt
imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@main
```

### Step 4: Add Benchmark-Specific Dependencies (if needed)

If your benchmark needs additional system packages, add them to the `apt-get install` line:

```dockerfile
RUN apt-get update && apt-get install -y \
    build-essential \
    your-package-here \
    && rm -rf /var/lib/apt/lists/*
```

## Example: INQUIRE Dockerfiles

See `benchmarks/INQUIRE/` for complete examples:

- **Dockerfile.benchmark**: Runs `main.py` (evaluator)
- **Dockerfile.data_loader**: Runs `load_data.py` (data loader)

Both follow the template pattern and only differ in the `CMD` line.

## Template Structure

```dockerfile
# Benchmark Dockerfile Template
ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Note: The imsearch-eval package is installed via requirements.txt
# No need to set PYTHONPATH as the package is installed in the Python environment

# Set your entrypoint script here
CMD ["python", "main.py"]
```

## Customization Options

### Python Version

Override the Python version via build arg:

```bash
docker build --build-arg PYTHON_VERSION=3.10-slim -t myimage .
```

Or set it in the Dockerfile:

```dockerfile
ARG PYTHON_VERSION=3.10-slim
```

### Additional System Packages

Add packages to the `apt-get install` command:

```dockerfile
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
```

### Custom Dependencies

If your benchmark needs additional Python packages, add them to `requirements.txt`:

```txt
imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@main
your-custom-package>=1.0.0
```

## Best Practices

1. **Keep it Simple**: Only customize what's necessary for your benchmark
2. **Follow the Template**: Use the template as a starting point to maintain consistency
3. **Document Changes**: Add comments explaining any customizations
4. **Test Locally**: Build and test your Dockerfiles before deploying
5. **Version Requirements**: Pin Python package versions in `requirements.txt` for reproducibility

## Integration with Makefile

The Makefile system automatically uses your Dockerfiles:

- `DOCKERFILE_EVALUATOR` (default: `Dockerfile.benchmark`)
- `DOCKERFILE_DATA_LOADER` (default: `Dockerfile.data_loader`)

Set these in your benchmark's Makefile:

```makefile
DOCKERFILE_EVALUATOR := Dockerfile.benchmark
DOCKERFILE_DATA_LOADER := Dockerfile.data_loader
```

## Troubleshooting

### Build Fails: "requirements.txt not found"

Ensure `requirements.txt` exists in your benchmark directory.

### Import Errors: "No module named 'imsearch_eval'"

Ensure `requirements.txt` includes the `imsearch-eval` package:
```txt
imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@main
```

Verify the package is installed by checking the build logs or running:
```bash
docker run <your-image> pip list | grep imsearch-eval
```

### Command Not Found: "python: command not found"

Ensure the Python base image is correct. The template uses `python:3.11-slim`.

## Benefits

- **Consistency**: All benchmarks use the same base setup
- **Maintainability**: Update the template to fix issues for all benchmarks
- **Flexibility**: Each benchmark can customize as needed
- **Documentation**: Template serves as documentation for the structure

