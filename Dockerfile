# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY static/ ./static/
COPY entrypoint.sh ./entrypoint.sh

# Create non-root user with home directory
RUN useradd --create-home --shell /bin/false appuser

# Create all directories the app needs and give appuser ownership
RUN mkdir -p /app/chroma_db /app/.cache/huggingface && \
    chown -R appuser:appuser /app && \
    chmod +x /app/entrypoint.sh

# Tell HuggingFace where to cache models
ENV HF_HOME=/app/.cache/huggingface

USER appuser

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
