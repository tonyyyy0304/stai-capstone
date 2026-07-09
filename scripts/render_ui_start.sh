#!/bin/sh
set -eu

: "${API_URL:?Set API_URL to your deployed FastAPI service URL.}"

PUBLIC_PORT="${PORT:-8501}"

echo "Starting Streamlit on 0.0.0.0:${PUBLIC_PORT}, using API_URL=${API_URL}..."
exec streamlit run src/ui.py \
  --server.address 0.0.0.0 \
  --server.port "$PUBLIC_PORT" \
  --server.headless true
