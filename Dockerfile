# XPLIA Production Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY setup.py README.md ./
COPY xplia ./xplia/

# Install XPLIA with full dependencies
RUN pip install --no-cache-dir --user -e ".[full]"

# Stage 2: Runtime
FROM python:3.10-slim

LABEL maintainer="Nicolas Severino <contact@xplia.com>"
LABEL description="XPLIA - The Ultimate State-of-the-Art AI Explainability Library"
LABEL version="1.0.0"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /build /app

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Create directories for data and outputs
RUN mkdir -p /data /outputs /models

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import xplia; print(xplia.__version__)" || exit 1

# Default command: start API server
CMD ["uvicorn", "xplia.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
