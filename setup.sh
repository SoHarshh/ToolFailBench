#!/usr/bin/env bash
set -e

echo "==> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> Creating virtual environment with uv..."
uv venv --python 3.13

echo "==> Installing Python dependencies..."
uv pip install -r requirements.txt

echo "==> Installing Claude Code..."
curl -fsSL https://claude.ai/install.sh | bash

echo ""
echo "============================================"
echo "Setup complete! Activate your venv with:"
echo ""
echo "  source .venv/bin/activate"
echo "============================================"
