#!/usr/bin/env bash
# One-time environment setup for the GLM-5.3-Flash EXL3 serving stack.
#
# Installs a Python venv with ExLlamaV3 1.5.4 (built for SM80) and the API
# dependencies. Model weights are NOT downloaded here; scripts/serve.sh
# fetches them from Hugging Face on first run.
#
# Requirements: CUDA toolkit with nvcc (SM80 / compute_80), ~10 GB build disk.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${GLM53_VENV:-$REPO_ROOT/.venv}"
ARCH="${GLM53_CUDA_ARCH:-8.0}"

cd "$REPO_ROOT"

if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating venv at $VENV"
  python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip

echo "== installing PyTorch (CUDA build) + API deps"
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu130
"$PY" -m pip install -r requirements.txt

echo "== building ExLlamaV3 1.5.4 (SM80 native kernels)"
TORCH_CUDA_ARCH_LIST="$ARCH" "$PY" -m pip install --no-build-isolation "exllamav3==1.5.4"

"$PY" - <<'EOF'
import torch, exllamav3
print("torch", torch.__version__, "| exllamav3", exllamav3.__version__ if hasattr(exllamav3, "__version__") else "1.5.4")
print("cuda available:", torch.cuda.is_available())
EOF

echo
echo "Setup complete. Next: scripts/serve.sh --profile q8_384k"
