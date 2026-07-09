#!/bin/sh
set -eu

MLFLOW_PORT="${PORT:-5000}"
MLFLOW_WORKERS="${MLFLOW_WORKERS:-1}"
mkdir -p data/mlruns

echo "Starting MLflow on 0.0.0.0:${MLFLOW_PORT} with ${MLFLOW_WORKERS} worker(s)..."
exec mlflow server \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT" \
  --workers "$MLFLOW_WORKERS" \
  --backend-store-uri /app/data/mlruns
