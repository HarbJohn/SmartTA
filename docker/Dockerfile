# syntax=docker/dockerfile:1.4
# SmartTA Docker Image - Multi-stage build with baked models
# Build time: ~15-20min, Runtime start: <10s

# Stage 1: Install dependencies
FROM python:3.11-slim AS builder

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl

WORKDIR /build

COPY requirements-minimal.txt ./

# Install dependencies with CPU-optimized PyTorch
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Platform-aware PyTorch installation
ARG TARGETPLATFORM
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TARGETPLATFORM" = "linux/arm64" ] || [ "$TARGETPLATFORM" = "linux/aarch64" ]; then \
      pip install --prefix=/install \
        torch==2.9.0 torchvision torchaudio \
        numpy scipy pandas scikit-learn threadpoolctl joblib faiss-cpu \
        transformers sentence-transformers huggingface-hub safetensors tokenizers \
        streamlit altair plotly matplotlib pillow wordcloud python-dateutil tenacity \
        rank-bm25 yt-dlp openai-whisper openai python-dotenv psutil requests regex; \
    else \
      pip install --prefix=/install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.9.0+cpu torchvision torchaudio \
        numpy scipy pandas scikit-learn threadpoolctl joblib faiss-cpu \
        transformers sentence-transformers huggingface-hub safetensors tokenizers \
        streamlit altair plotly matplotlib pillow wordcloud python-dateutil tenacity \
        rank-bm25 yt-dlp openai-whisper openai python-dotenv psutil requests regex; \
    fi \
    && echo "✅ Packages installed for $TARGETPLATFORM"

# Verify torch installation
ARG TARGETPLATFORM
RUN PYTHONPATH=/install/lib/python3.11/site-packages python -c "import torch; print('Platform: $TARGETPLATFORM'); print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); assert not torch.cuda.is_available(), 'CUDA detected - build failed!'" \
    && echo "✅ Torch verification passed for $TARGETPLATFORM"

# Stage 2: Copy models from host
FROM scratch AS model-copier
COPY data/models /models

# Stage 3: Runtime image
FROM python:3.11-slim AS runtime

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r smartta -g 10001 && useradd -r -g smartta -u 10001 smartta

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=model-copier --chown=smartta:smartta /models /app/data/models

# Environment variables
ENV HF_HOME=/app/data/models \
    SENTENCE_TRANSFORMERS_HOME=/app/data/models \
    TORCH_HOME=/app/data/models \
    WHISPER_MODELS_DIR=/app/data/models \
    TRANSFORMERS_CACHE=/app/data/models \
    HF_HUB_ENABLE_PROGRESS_BARS=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    FONTCONFIG_PATH=/etc/fonts

RUN echo "✓ Models copied from host" \
    && ls -lh /app/data/models/ || echo "⚠️ Models directory empty"

# Copy application code
COPY --chown=smartta:smartta .streamlit/ .streamlit/
COPY --chown=smartta:smartta SmartTA_RAG/ SmartTA_RAG/
COPY --chown=smartta:smartta SmartTA_Extensions/ SmartTA_Extensions/

# Copy shared data files
COPY --chown=smartta:smartta data/course_feedback.csv data/course_feedback.csv
COPY --chown=smartta:smartta data/chat_logs.ndjson data/chat_logs.ndjson
COPY --chown=smartta:smartta data/gold_queries.json data/gold_queries.json

# Copy entrypoint script
COPY --chown=root:root docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /app/SmartTA_Extensions/data/lectures /app/SmartTA_Extensions/data/transcripts \
    /app/SmartTA_Extensions/data/uploaded_files \
    /app/SmartTA_RAG/data/index \
    /app/data-metadata /app/logs /tmp/matplotlib \
    && chown -R smartta:smartta /app/SmartTA_Extensions/data /app/SmartTA_RAG/data /app/data /app/logs /tmp/matplotlib

USER smartta
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit", "run", "/app/SmartTA_RAG/smartta_unified/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
