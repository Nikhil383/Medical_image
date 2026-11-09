FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency definitions and lockfile
COPY pyproject.toml .python-version ./

# Install dependencies using uv
RUN uv venv .venv && \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install monai pydicom fastapi uvicorn requests scikit-learn polars matplotlib tqdm dvc pytest httpx python-multipart

# Copy source code and frontend
COPY src/ /app/src/
COPY frontend/ /app/frontend/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/

RUN uv pip install -e . --no-deps

EXPOSE 8000

ENV PYTHONPATH=/app/src
ENV PORT=8000
CMD ["sh", "-c", ".venv/bin/uvicorn medical_image.api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
