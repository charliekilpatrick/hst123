#!/usr/bin/env bash
# Run hst123 for SN 2021tyw region in test_data/ (download, drizzle-all, dolphot, scrape).
#
# Prerequisites (from repo root):
#   pip install -e .
#   # plus drizzlepac; DOLPHOT binaries on PATH for --run-dolphot
#
# Requires: network to MAST for --download and get_productlist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
# Prefer env Python (conda/venv) so we don't hit macOS CLT python3 (no hst123).
# Override with PYTHON=... if needed.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
  fi
fi
exec "${PYTHON:-python3}" -m hst123 346.48525 14.3577 \
  --work-dir test_data \
  --download \
  --drizzle-all \
  --scrape-radius 3.0 \
  --after 2022-11-10 \
  --run-dolphot \
  --scrape-dolphot \
  "$@"
