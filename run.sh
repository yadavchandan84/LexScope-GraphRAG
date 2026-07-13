#!/usr/bin/env bash
#
# One-shot bootstrap + run for the Legal RAG system.
#
# From a clean checkout this will:
#   1. create a Python virtualenv (.venv)
#   2. install all Python dependencies
#   3. ingest the PDFs  -> ingestion/chunks.json
#   4. index the chunks -> embedded Qdrant (dense+sparse) + Neo4j citation graph
#   5. start the FastAPI backend (port 8000) and the frontend (port 5500)
#
# No local database setup is required: Qdrant runs embedded on disk, and Neo4j +
# Gemini are cloud services read from .env. The ONLY prerequisites are:
#   - Python 3.10+ on PATH
#   - a filled-in .env (copy .env.example and add real credentials)
#
# Works on Linux/macOS and on Windows via Git Bash.
#
# Usage:
#   ./run.sh                # full bootstrap + ingest + index + serve
#   ./run.sh --skip-index   # bootstrap + serve only (reuse existing indexes)
#   ./run.sh --no-serve     # bootstrap + ingest + index, then exit (no servers)

set -euo pipefail
cd "$(dirname "$0")"

API_PORT=8000
FRONTEND_PORT=5500
SKIP_INDEX=0
NO_SERVE=0
for arg in "$@"; do
  case "$arg" in
    --skip-index) SKIP_INDEX=1 ;;
    --no-serve)   NO_SERVE=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

# --- locate a Python interpreter ---------------------------------------------
if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "ERROR: Python 3.10+ not found on PATH." >&2
  exit 1
fi

# --- create venv -------------------------------------------------------------
if [ ! -d .venv ]; then
  echo ">> Creating virtualenv (.venv)"
  "$PYTHON" -m venv .venv
fi

# venv layout differs between Windows (Scripts) and POSIX (bin)
if [ -x ".venv/Scripts/python.exe" ]; then
  VPY=".venv/Scripts/python.exe"
else
  VPY=".venv/bin/python"
fi

# --- install dependencies ----------------------------------------------------
echo ">> Installing dependencies (this is slow the first time)"
"$VPY" -m pip install --upgrade pip -q
"$VPY" -m pip install -r requirements.txt -q

# --- check .env --------------------------------------------------------------
if [ ! -f .env ]; then
  echo "ERROR: .env not found. Copy .env.example to .env and fill in credentials." >&2
  exit 1
fi

# --- ingest + index ----------------------------------------------------------
# Each step is its own process, so the embedded Qdrant folder lock is released
# before the API server acquires it.
if [ "$SKIP_INDEX" -eq 0 ]; then
  echo ">> Ingesting PDFs -> ingestion/chunks.json"
  "$VPY" -m ingestion.run_ingest

  echo ">> Indexing chunks -> Qdrant (embedded) + Neo4j (first run downloads BGE models, ~2.5GB)"
  "$VPY" -m indexing.run_index
else
  echo ">> Skipping ingest/index (--skip-index)"
fi

if [ "$NO_SERVE" -eq 1 ]; then
  echo ">> Done (--no-serve). Indexes are built."
  exit 0
fi

# --- serve -------------------------------------------------------------------
echo ">> Starting FastAPI backend on http://localhost:${API_PORT}"
"$VPY" -m uvicorn app.api:app --port "$API_PORT" &
API_PID=$!

echo ">> Starting frontend on http://localhost:${FRONTEND_PORT}"
( cd frontend && "../$VPY" -m http.server "$FRONTEND_PORT" ) &
FRONTEND_PID=$!

cleanup() {
  echo ""
  echo ">> Shutting down..."
  kill "$API_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "==================================================================="
echo "  Legal RAG is up:"
echo "    API   -> http://localhost:${API_PORT}/health"
echo "    UI    -> http://localhost:${FRONTEND_PORT}"
echo "  Press Ctrl+C to stop."
echo "==================================================================="

# wait on the backend; if it dies, tear everything down
wait "$API_PID"
