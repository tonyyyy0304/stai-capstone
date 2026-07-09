#!/bin/sh
set -eu

export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${PORT:-8000}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:///app/data/mlruns}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-hr-agent}"
export ENABLE_MLFLOW="${ENABLE_MLFLOW:-0}"
export RUN_INGEST_ON_START="${RUN_INGEST_ON_START:-auto}"

mkdir -p data

if [ "$RUN_INGEST_ON_START" = "0" ]; then
  echo "Skipping ingestion because RUN_INGEST_ON_START=0."
elif [ "$RUN_INGEST_ON_START" = "auto" ] \
  && [ -f data/index_manifest.json ] \
  && [ -f data/chroma/chroma.sqlite3 ]; then
  echo "Existing Chroma index found; skipping startup ingestion."
else
  echo "Preparing knowledge base..."
  python scripts/ingest.py
fi

echo "Starting FastAPI on ${API_HOST}:${API_PORT}..."
exec uvicorn src.api:app --host "$API_HOST" --port "$API_PORT"
