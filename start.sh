#!/bin/bash
set -e
PORT="${PORT:-8000}"
echo "Starting TradingAgents Web on 0.0.0.0:$PORT"
exec python -m uvicorn web.server:app --host 0.0.0.0 --port "$PORT"
