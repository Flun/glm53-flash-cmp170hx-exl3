"""DFlash2 -> EXL3 conversion driver (MiaAI-Lab/exllamav3 @ 63b32f0, v1.4.2, MIT).

Wraps the stock converter CLI (exllamav3/conversion/convert_model.py) with two
process-local adjustments; the fork tree itself is not modified:

1. k_proj/v_proj stay BF16: the q-strategy entries for those keys are set to 16
   (the converter's "unquantized" sentinel, see convert_model.py:441). Serving
   builds a fused context-KV weight from the qkv projection's K/V rows
   (vLLM qwen3_dflash.py _build_context_kv_buffers), which must remain a plain
   tensor, so k/v are excluded from quantization by design.
2. DFlash2DynConv's kernel_projection Linears get a qmap so they enter the
   quantization budget. The fork constructs them with qmap=None (never
   quantized); the recipe quantizes q, o, gate, up, down, kernel_projection
   and fc. Set env DRAFT_QUANT_KERNEL_PROJ=0 to leave kernel_projection
   unquantized — with qmap=None the converter never touches them, the
   cleanest K16 equivalent. Set DRAFT_QUANT_FC=0 to keep fc (the 5-layer
   target-hidden input projection) BF16: fc carries
   qmap="target_hidden.input", so the K16 sentinel is the mechanism there.

candidate_selector.hidden_projection keeps the fork default (qmap=None,
unquantized): it is 1M params and steers the selector — not worth 2 MB.

DFlash2 models carry caps["uncalibrated_quantize"] = True, so the converter
skips calibration forwards entirely (state=None) and quantizes each linear
against the synthetic Hessian (init_H_data). No target model or calibration
text is involved; -cr/-cc are irrelevant for this arch.
"""

import os
import sys

_QUANT_KP = os.environ.get("DRAFT_QUANT_KERNEL_PROJ", "1") == "1"
_QUANT_FC = os.environ.get("DRAFT_QUANT_FC", "1") == "1"

import exllamav3.modules.arch_specific.dflash2 as _d2

_orig_dynconv_init = _d2.DFlash2DynConv.__init__


def _dynconv_init(self, config, key, hidden_size, kernel_size, group_size, qmap=None):
    _orig_dynconv_init(
        self, config, key, hidden_size, kernel_size, group_size,
        qmap=qmap if qmap is not None else "dynconv",
    )


if _QUANT_KP:
    _d2.DFlash2DynConv.__init__ = _dynconv_init
print(f" -- [draft-quant] kernel_projection quantization: {_QUANT_KP}, fc: {_QUANT_FC}")

import exllamav3.conversion.convert_model as _cm

_orig_create_q_strategy = _cm.create_q_strategy
_BF16_KEYS = (".k_proj", ".v_proj")


def _create_q_strategy(*args, **kwargs):
    strategy, final_bpw = _orig_create_q_strategy(*args, **kwargs)
    kept = [k for k in list(strategy) if k.endswith(_BF16_KEYS)]
    if not _QUANT_FC:
        # fc (the 5-layer target-hidden input projection) carries
        # qmap="target_hidden.input", so the K16 sentinel is the mechanism
        # (unlike kernel_projection's qmap skip).
        assert "fc" in strategy
        strategy["fc"] = 16
        kept.append("fc")
    for key in kept:
        strategy[key] = 16
    names = sorted(k.rsplit(".", 2)[-2] if "." in k else k for k in kept)
    print(f" -- [draft-quant] keeping BF16: {len(kept)} tensors ({', '.join(names)})")
    return strategy, final_bpw


_cm.create_q_strategy = _create_q_strategy

if __name__ == "__main__":
    _args = _cm.parser.parse_args()
    _in_args, _job_state, _ok, _err = _cm.prepare(_args)
    if not _ok:
        print(f" !! Error: {_err}")
        sys.exit(1)
    _cm.main(_in_args, _job_state)
