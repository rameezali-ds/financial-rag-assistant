FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for lxml and unstructured
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY . .

# Pre-download embedding model to avoid cold start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

EXPOSE 8000 8501

ENV CHROMA_PERSIST_DIR=/data/chroma

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
