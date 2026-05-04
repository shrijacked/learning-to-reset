#!/usr/bin/env bash
# Optional OS packages on Debian/Ubuntu (needs root). Safe no-op elsewhere.
# Fresh ubuntu:22.04 images often lack python3-venv; this installs it when possible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[bootstrap_ubuntu] repo root: ${REPO_ROOT}"

if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    || {
      echo "[bootstrap_ubuntu] WARNING: apt-get install failed; ensure python3 and python3-venv exist." >&2
    }
elif command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    || {
      echo "[bootstrap_ubuntu] WARNING: sudo apt-get failed; ensure python3-venv is installed." >&2
    }
else
  echo "[bootstrap_ubuntu] skipping apt (not root / no passwordless sudo). Ensure python3 >= 3.9 and python3-venv are installed."
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[bootstrap_ubuntu] ERROR: python3 not found." >&2
  exit 1
fi

python3 --version
