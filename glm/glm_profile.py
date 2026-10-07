"""Validated full-VRAM GLM profile shared by the dashboard and text API."""
MODEL_ID = "GLM-5.3-Flash"
PORT = 8012
GENERATION_DEFAULTS = dict(reasoning_effort="high", temperature=1.0, top_p=0.95,
                           max_tokens=32768, frequency_penalty=0.0, presence_penalty=0.0)
MODEL_CONTEXT_LIMIT = 1048576
# Reserve one page for the speculative window at the request boundary.
CONTEXT_RESERVE = 256
PROFILES = {
    "fp16": dict(label="192K · FP16", cache_size=196608, cache_quant=None,
                 staging=False, head_gpu0=False, margin_mb=256, validated=True, note="기존 속도 우선 설정"),
    "q8_fast": dict(label="256K · Q8 · 속도 우선", cache_size=262144, cache_quant=8,
                    staging=True, head_gpu0=False, margin_mb=192, validated=True, mtp_gpu_split="59,63",
                    note="DFlash2 K7 · target Q8 · drafter GPU 링 캐시 8K"),
    "q8_320k": dict(label="320K · Q8", cache_size=327680, cache_quant=8,
                    staging=True, head_gpu0=False, margin_mb=256, validated=True,
                    note="DFlash2 K7 · 확장 컨텍스트"),
    "q8_384k": dict(label="384K · Q8 · 확장 설정", cache_size=393216, cache_quant=8,
                    staging=True, head_gpu0=True, margin_mb=128, validated=True,
                    note="DFlash2 K7 · 출력 레이어 GPU 0 · 메모리 여유 적음"),
}
DEFAULT_PROFILE = "fp16"
DEFAULT = PROFILES[DEFAULT_PROFILE]
CACHE_SIZE = DEFAULT["cache_size"]
CACHE_QUANT = DEFAULT["cache_quant"]
Q8_STAGING = DEFAULT["staging"]
CACHE_FORMAT = "FP16" if CACHE_QUANT is None else f"Q{CACHE_QUANT}"
CONTEXT_LENGTH = min(MODEL_CONTEXT_LIMIT, CACHE_SIZE - CONTEXT_RESERVE)
GPU_SPLIT = "62,63"
