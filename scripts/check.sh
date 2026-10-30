#!/usr/bin/env bash
# Local check: lint, tests, CLI smoke. Creates .venv with uv on first run.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONDONTWRITEBYTECODE=1

if ! command -v uv >/dev/null 2>&1; then
    echo "check.sh: uv not found. Install it from https://docs.astral.sh/uv/" >&2
    exit 2
fi
[ -x .venv/bin/python ] || uv venv -q -p 3.11 .venv
uv pip install -q -p .venv/bin/python -e '.[dev]'
py=.venv/bin/python

echo "== ruff"
"$py" -m ruff check --no-cache .
echo "== pytest"
"$py" -m pytest tests -q -p no:cacheprovider
echo "== simsteer --help"
"$py" -m simsteer --help
