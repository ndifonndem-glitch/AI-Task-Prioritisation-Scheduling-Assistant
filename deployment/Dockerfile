# IT8532 Assessment 2 - Proxmox CT 200 Deployment Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY ml_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and trained model artifacts
COPY data/ /app/data/
COPY ml_service/ /app/ml_service/
COPY agent/ /app/agent/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose FastAPI server port
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Launch uvicorn web server
CMD ["uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "5000"]
