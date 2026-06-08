# Stage 1: Build the virtual environment
FROM python:3.11-slim AS builder

# Install uv
RUN pip install --no-cache-dir uv

# Set environment variables for uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Set the working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Sync dependencies (excluding dev dependency group for production stage)
# We also omit installing the project itself in this step to maximize caching
RUN uv sync --frozen --no-install-project --no-dev


# Stage 2: Final runner stage
FROM python:3.11-slim AS runner

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Install curl for healthchecks and CA certs, then clean up
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root group and user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -d /app -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Create log directory and set proper permissions
RUN mkdir -p /app/logs && chown -R appuser:appgroup /app/logs

# Copy the virtual environment from builder stage
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Copy the rest of the application files
COPY --chown=appuser:appgroup . /app

# Switch to the non-root user
USER appuser

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000
EXPOSE 8501

# Secure healthcheck targeting FastAPI's health endpoint using curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default runtime command (FastAPI Backend)
# This can be overridden in docker-compose for the Streamlit Frontend service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
