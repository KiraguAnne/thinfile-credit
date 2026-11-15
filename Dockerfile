# Model-serving image: FastAPI + the trained LightGBM artifact.
# Build after `make train` so artifacts/models/ contains the model:
#   docker compose up --build
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

# LightGBM needs the OpenMP runtime, which the slim image does not ship.
ARG APT_PACKAGES="libgomp1"
RUN if [ -n "$APT_PACKAGES" ]; then \
      apt-get update && apt-get install -y --no-install-recommends $APT_PACKAGES && \
      rm -rf /var/lib/apt/lists/*; \
    fi

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app \
    MODEL_PATH=/app/artifacts/models/lgbm_C.joblib
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY api/ api/
COPY configs/ configs/
COPY artifacts/models/ artifacts/models/
COPY artifacts/table_schemas.json artifacts/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
