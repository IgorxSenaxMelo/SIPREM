#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "Criando ambiente Python (primeira execução)..."
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt || exit 1
fi
.venv/bin/streamlit run app.py
