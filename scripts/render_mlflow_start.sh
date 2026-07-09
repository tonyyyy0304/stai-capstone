#!/bin/sh
set -eu

MLFLOW_PORT="${PORT:-5000}"
mkdir -p data/mlruns

echo "Starting MLflow on 0.0.0.0:${MLFLOW_PORT}..."
exec mlflow server \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT" \
  --backend-store-uri /app/data/mlruns
