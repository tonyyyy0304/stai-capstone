#!/bin/sh
set -eu

API_PORT="${API_PORT:-8000}"
PUBLIC_PORT="${PORT:-8501}"

export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT
export API_URL="${API_URL:-http://127.0.0.1:${API_PORT}}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-hr-agent}"

mkdir -p data

echo "Preparing knowledge base..."
python scripts/ingest.py

echo "Starting MLflow on 127.0.0.1:5000..."
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri /app/data/mlruns &
MLFLOW_PID=$!

echo "Starting FastAPI on ${API_HOST}:${API_PORT}..."
uvicorn src.api:app --host "$API_HOST" --port "$API_PORT" &
API_PID=$!

cleanup() {
  kill "$API_PID" "$MLFLOW_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "Starting Streamlit on 0.0.0.0:${PUBLIC_PORT}..."
streamlit run src/ui.py \
  --server.address 0.0.0.0 \
  --server.port "$PUBLIC_PORT" \
  --server.headless true
