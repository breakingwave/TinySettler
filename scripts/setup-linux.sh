#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(uname -s) == Linux && -f /.dockerenv ]] || { echo 'Run inside the Linux coding container.' >&2; exit 1; }
[[ $(node -p 'process.versions.node.split(".")[0]') == 24 ]] || { echo 'Node 24 required.' >&2; exit 1; }
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
if [[ ! -e .venv ]]; then uv venv --python python3 .venv; fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing venv is not Python 3.12; leaving it intact"'
uv pip sync --python .venv/bin/python backend/requirements-lock.txt
npm ci --no-audit --no-fund
PYTHONHASHSEED=0 .venv/bin/python -c 'import catanatron; import backend.catan_server; print("Backend imports OK")'
