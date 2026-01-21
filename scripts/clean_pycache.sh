#!/bin/bash
# Delete all __pycache__ directories in the project (excluding .venv)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Cleaning __pycache__ directories in: $PROJECT_ROOT"
echo "Excluding: .venv"
echo

# Find and delete __pycache__ directories, excluding .venv
find "$PROJECT_ROOT" -type d -name ".venv" -prune -o -type d -name "__pycache__" -print -exec rm -rf {} + 2>/dev/null || true

echo "Done."
