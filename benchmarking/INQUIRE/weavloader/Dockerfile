FROM python:3.11-slim

RUN apt-get update \
  && apt-get install -y \
  wget \
  curl

# Set working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Run upload.py
CMD ["python", "main.py"]