#!/usr/bin/env bash
# Serve GLM-5.3-Flash EXL3 on 2x 64GB GPUs (CMP 170HX / SM80 class).
#
# First run downloads the target EXL3 3.05bpw checkpoint and the BF16 DFlash2
# drafter from Hugging Face, then converts the drafter to EXL3 6bpw. Both land
# under --models-dir (default ./models) and are reused afterwards.
#
# Profiles (validated memory layouts on 2x64GB):
#   q8_384k  (default)  384K Q8 KV, drafter ring cache, lm_head on GPU 0
#   q8_320k             320K Q8
#   q8_fast             256K Q8
#   fp16                192K FP16
#   mtp                 MTP d2 speculation instead of DFlash2 K7
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${GLM53_MODELS_DIR:-$REPO_ROOT/models}"
VENV="${GLM53_VENV:-$REPO_ROOT/.venv}"
PY="$VENV/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

TARGET_REPO="turboderp/GLM-5.3-Flash-exl3"
TARGET_REV="332ab457b709b7ba30dd9a448be5de03b80a7ac9"   # 3.05bpw branch tip
TARGET_DIR="$MODELS_DIR/GLM-5.3-Flash-exl3-3.05bpw"
DRAFT_BF16_REPO="incoai/GLM-5.3-Flash-DFlash2"
DRAFT_BF16_REV="bf582e4eacc1810f76656d1811693ff6c6737d2a"
DRAFT_BF16_DIR="$MODELS_DIR/GLM-5.3-Flash-DFlash2-bf16"
DRAFT_EXL3_DIR="$MODELS_DIR/GLM-5.3-Flash-DFlash2-exl3-6bpw"

PROFILE="q8_384k"
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; TARGET_DIR="$2/GLM-5.3-Flash-exl3-3.05bpw";
                  DRAFT_BF16_DIR="$2/GLM-5.3-Flash-DFlash2-bf16";
                  DRAFT_EXL3_DIR="$2/GLM-5.3-Flash-DFlash2-exl3-6bpw"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$MODELS_DIR"

hf_download() {  # repo revision out_dir
  local repo="$1" rev="$2" out="$3"
  if [ -f "$out/model.safetensors.index.json" ] || [ -f "$out/model.safetensors" ]; then
    echo "== already present: $out"
    return 0
  fi
  echo "== downloading $repo@$rev -> $out"
  "$PY" -m huggingface_hub.commands.huggingface_cli download "$repo" \
    --revision "$rev" --local-dir "$out"
}

hf_download "$TARGET_REPO" "$TARGET_REV" "$TARGET_DIR"

SPEC_ARGS=()
if [ "$PROFILE" = "mtp" ]; then
  SPEC_ARGS=(--mtp -ndt 2)
else
  hf_download "$DRAFT_BF16_REPO" "$DRAFT_BF16_REV" "$DRAFT_BF16_DIR"
  if [ ! -f "$DRAFT_EXL3_DIR/model.safetensors" ]; then
    echo "== converting DFlash2 drafter to EXL3 6bpw (one-time, ~tens of minutes)"
    mkdir -p "$MODELS_DIR/dflash2-quant-work"
    DRAFT_QUANT_KERNEL_PROJ=1 DRAFT_QUANT_FC=1 \
      "$PY" "$REPO_ROOT/dflash2/convert_drafter.py" \
        -i "$DRAFT_BF16_DIR" -o "$DRAFT_EXL3_DIR" \
        -w "$MODELS_DIR/dflash2-quant-work" -b 6 --devices 0
    "$PY" "$REPO_ROOT/dflash2/package_native.py" \
      -i "$DRAFT_BF16_DIR" -o "$DRAFT_EXL3_DIR"
  fi
  SPEC_ARGS=(-dm "$DRAFT_EXL3_DIR" -ndt 7)
fi

case "$PROFILE" in
  q8_384k) MEM_ARGS=(-cs 393216 -cq 8 -gs 62,63 --q8-staging --q8-head-gpu0) ;;
  q8_320k) MEM_ARGS=(-cs 327680 -cq 8 -gs 62,63 --q8-staging) ;;
  q8_fast) MEM_ARGS=(-cs 262144 -cq 8 -gs 62,63 --q8-staging) ;;
  fp16)    MEM_ARGS=(-cs 196608 -gs 62,63) ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac

# Autosplit loader margin (EXL3_AUTOSPLIT_MARGIN_MB); 384K needs the tightest
# budget with the output projection reserved on GPU 0.
case "$PROFILE" in
  q8_384k) MARGIN_MB=128 ;;
  q8_fast) MARGIN_MB=192 ;;
  *)       MARGIN_MB=256 ;;
esac

PORT="${GLM53_PORT:-8012}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export GLM53_K_HCFUSE=1
export EXL3_AUTOSPLIT_MARGIN_MB="$MARGIN_MB"
export TORCH_CUDA_ARCH_LIST="${GLM53_CUDA_ARCH:-8.0}"   # first k_hcfuse JIT build

echo "== starting GLM API on port $PORT (profile $PROFILE)"
exec "$PY" -u "$REPO_ROOT/glm/glm_api.py" \
  -m "$TARGET_DIR" "${MEM_ARGS[@]}" "${SPEC_ARGS[@]}" \
  -chunk_size 2048 --autosplit_max_batch_size 1 --port "$PORT" \
  "${EXTRA_ARGS[@]}"
