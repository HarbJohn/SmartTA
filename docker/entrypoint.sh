#!/usr/bin/env bash
set -euo pipefail

# SmartTA Container Entrypoint
# Handles metadata persistence, API key loading, and model verification

log() { printf '%s\n' "[Entrypoint] $*"; }
warn() { printf '%s\n' "[Entrypoint][WARN] $*"; }
err() { printf '%s\n' "[Entrypoint][ERROR] $*" >&2; }

graceful_shutdown() {
    warn "Shutting down..."
    sleep 2
}
trap graceful_shutdown SIGTERM SIGINT

# Persist metadata via symlinks
EXT_METADATA_FILES=(
    "metadata.json"
    "segments_metadata.json"
    "course.index"
    "queries_log.json"
    "course_materials.json"
)

RAG_METADATA_FILES=(
    "course_feedback.csv"
    "chat_logs.ndjson"
    "gold_queries.json"
)

mkdir -p /app/data-metadata || err "Failed to create /app/data-metadata"

if touch /app/data-metadata/.writable-test 2>/dev/null; then
    rm -f /app/data-metadata/.writable-test
    for file in "${EXT_METADATA_FILES[@]}"; do
        SOURCE="/app/SmartTA_Extensions/data/$file"
        TARGET="/app/data-metadata/ext_$file"
        if [ -f "$TARGET" ]; then
            ln -sf "$TARGET" "$SOURCE" 2>/dev/null || true
        elif [ -f "$SOURCE" ]; then
            mv "$SOURCE" "$TARGET" 2>/dev/null && ln -sf "$TARGET" "$SOURCE" 2>/dev/null || true
        else
            touch "$TARGET" 2>/dev/null && ln -sf "$TARGET" "$SOURCE" 2>/dev/null || true
        fi
    done

    for file in "${RAG_METADATA_FILES[@]}"; do
        SOURCE="/app/data/$file"
        TARGET="/app/data-metadata/$file"
        if [ -f "$TARGET" ]; then
            ln -sf "$TARGET" "$SOURCE" 2>/dev/null || true
        elif [ -f "$SOURCE" ]; then
            mv "$SOURCE" "$TARGET" 2>/dev/null && ln -sf "$TARGET" "$SOURCE" 2>/dev/null || true
        else
            touch "$TARGET" 2>/dev/null && ln -sf "$TARGET" "$SOURCE" 2>/dev/null || true
        fi
    done
    log "Metadata persistence enabled (volume mounted)."
else
    log "Using ephemeral storage (data will not persist across restarts)."
fi

# Load API key
if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -f /run/secrets/openai_api_key ]; then
        export OPENAI_API_KEY="$(tr -d '\r' < /run/secrets/openai_api_key)"
        log "Loaded OPENAI_API_KEY from Docker Secret."
    elif [ -f /app/.env ]; then
        set -a
        . /app/.env
        set +a
        if [ -n "${OPENAI_API_KEY:-}" ]; then
            log "Loaded OPENAI_API_KEY from .env file."
        else
            warn ".env present but OPENAI_API_KEY missing."
        fi
    else
        warn "OPENAI_API_KEY not set. OpenAI-dependent features disabled."
    fi
fi

if [ -n "${OPENAI_API_KEY:-}" ]; then
  OPENAI_API_KEY="${OPENAI_API_KEY%$'\r'}"
  OPENAI_API_KEY="${OPENAI_API_KEY%\"}"
  OPENAI_API_KEY="${OPENAI_API_KEY#\"}"
  export OPENAI_API_KEY
  if [ ${#OPENAI_API_KEY} -lt 20 ]; then
    warn "OPENAI_API_KEY appears unusually short; check value."
  fi
fi

# Verify models
MODEL_ROOT="/app/data/models"
CLIP_DIR="$MODEL_ROOT/models--openai--clip-vit-large-patch14-336"
CE_DIR="$MODEL_ROOT/models--cross-encoder--ms-marco-MiniLM-L-6-v2"
missing_components=0
if [ -d "$CLIP_DIR" ]; then
    log "CLIP model detected."
else
    warn "CLIP model missing (minimal fallback may load)."
    missing_components=$((missing_components+1))
fi
if [ -d "$CE_DIR" ]; then
    log "Cross-Encoder detected (precision reranking enabled)."
else
    warn "Cross-Encoder missing (hybrid ranking only)."
    missing_components=$((missing_components+1))
fi

if [ $missing_components -gt 0 ]; then
    warn "$missing_components retrieval component(s) missing; system will degrade gracefully."
fi

MODEL_SIZE="$(du -sh /app/data/models 2>/dev/null | cut -f1 || echo '?')"
log "Model root size: $MODEL_SIZE"

for d in /app/data /app/SmartTA_Extensions/data /app/SmartTA_Extensions/data/lectures \
         /app/SmartTA_Extensions/data/transcripts /app/SmartTA_RAG/data/index; do
    if [ ! -d "$d" ]; then
        warn "Directory $d missing; creating..."
        mkdir -p "$d" 2>/dev/null || warn "Cannot create $d (may need volume mount)"
    fi
    if ! touch "$d/.wtest" 2>/dev/null; then
        warn "Directory $d not writable (ephemeral mode - changes won't persist)."
    else
        rm -f "$d/.wtest"
    fi
done

log "Starting Streamlit application..."
exec "$@"
