# Dockerfile — French-Learning-Perceptions ML (Level 2 MLOps)
# Image de serving pour l'API FastAPI

FROM python:3.13-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn python-multipart \
    && pip install --no-cache-dir spacy \
    && python -m spacy download fr_core_news_sm

# Code source
COPY src/ ./src/
COPY api/ ./api/
COPY params.yaml .
COPY models/ ./models/

# Port (configurable via API_PORT env var, default 8001)
ARG API_PORT=8001
ENV API_PORT=$API_PORT
EXPOSE $API_PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"API_PORT\",\"8001\")}/health')"

# Démarrage
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT:-8001}"]
