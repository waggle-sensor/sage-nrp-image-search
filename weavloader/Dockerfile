FROM python:3.11-slim

RUN apt-get update \
  && apt-get install -y \
  build-essential \
  wget \
  vim \
  jq \
  curl \
  redis-server \
  supervisor

# Set working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV MANIFEST_API=https://auth.sagecontinuum.org/manifests/
ENV CELERY_BROKER_URL=redis://localhost:6379/0
ENV CELERY_RESULT_BACKEND=redis://localhost:6379/0
ENV PYTHONUNBUFFERED=1
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc_dir

# Copy the rest of the application code into the container
COPY . .

# Create supervisor configuration
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create Redis data directory with proper permissions
RUN mkdir -p /var/lib/redis && chown -R root:root /var/lib/redis && chmod 755 /var/lib/redis

# Create Prometheus multiprocess directory
RUN mkdir -p /tmp/prometheus_multiproc_dir && chmod 755 /tmp/prometheus_multiproc_dir

# Expose ports
EXPOSE 8080

# Use supervisor to manage both Redis and Celery
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]