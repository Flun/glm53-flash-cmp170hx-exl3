# GLM-5.3-Flash EXL3 serving — 2× 64GB GPUs (CMP 170HX / SM80)

Full-VRAM serving stack for **GLM-5.3-Flash** with **DFlash2 K7 speculative
decoding**, tuned for two 64 GB GPUs with **no P2P support** (e.g. CMP 170HX
mining cards). It also runs stock **MTP d2** as an alternative speculation
mode. ExLlamaV3 sources are not modified; every engine adjustment here is a
process-local monkeypatch that is enabled only while this server runs.

English README (this file) · [한국어 README](README.ko.md)

## What is included

| Component | Path | Notes |
|---|---|---|
| OpenAI-compatible API server | `glm/glm_api.py` | chat/completions + SSE, native PP/TG timings, tool calls |
| DFlash2 integration patches | `glm/glm_dflash.py` | drafter pinning, 8K GPU ring cache, fused-export preservation |
| Responsive async generator | `glm/glm_async.py` | GPU steps off the HTTP event loop, shielded cancellation |
| Q8 memory layout patches | `glm/glm_q8_layout.py`, `glm/glm_q8_staging.py` | lm_head on GPU 0, bounded staging, batch-1 decode workspace |
| GLM tool-call translation | `glm/glm_tools.py` | native XML tool calls → OpenAI `tool_calls` |
| mHC fused decode kernels | `kernels/k_hcfuse.py` | 0xSero k_hcfuse, ExLlamaV3 1.5.4 compatible |
| DFlash2 BF16 → EXL3 6bpw conversion | `dflash2/convert_drafter.py`, `dflash2/package_native.py` | 36 quantized linears, ~0.96 GiB drafter |
| Setup / serve launchers | `scripts/setup.sh`, `scripts/serve.sh` | auto-download from Hugging Face on first run |

Not included (out of scope): the benchmark harness and raw measurement data
from the experiment campaign, and the dashboard integration that this server
was originally embedded in.

## Validated configuration

| Item | Value |
|---|---|
| Target model | `turboderp/GLM-5.3-Flash-exl3` @ `3.05bpw` (pinned revision) |
| Engine | ExLlamaV3 1.5.4 (SM80 native build) |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2` BF16 → self-converted EXL3 6bpw (~0.96 GiB) |
| Speculation | DFlash2 K7 (default) or MTP d2 |
| KV cache | Q8 (MLA latent), FP16 profile also available |
| Service context | 384K (Q8) / 320K / 256K / 192K (FP16) |
| Max request | input + output ≈ 392,960 tokens (384K profile) |
| GPUs | 2× 64 GB, SM80, no P2P (host-bounce D2D) |
| Split budget | `-gs 62,63` (each number is a per-component load budget, not a VRAM split) |

All weights stay resident in HBM: no CPU expert tier, no SSD paging, no
runtime weight transfer. This is a Strata-style HBM-first layout rebuilt on
ExLlamaV3: the memory that grows with context (KV cache, drafter state) is
kept small instead of offloading weights.

## Quick start

```bash
# 1. Environment (venv + ExLlamaV3 1.5.4 built for SM80 + API deps)
scripts/setup.sh

# 2. Serve. First run downloads the target (~125 GB) and the BF16 drafter
#    (~2 GB) from Hugging Face, then converts the drafter to EXL3 6bpw
#    (one-time, tens of minutes). Later runs reuse ./models.
scripts/serve.sh --profile q8_384k

# 3. Use it
curl http://127.0.0.1:8012/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

Requirements: CUDA toolkit with `nvcc` (compute_80), Linux, ~140 GB free disk
for models + conversion workspace, and both GPUs free of other workloads.
The server refuses to start if either GPU has less than ~64 GB free.

`HF_TOKEN` is not required for these public repos, but set it if you hit rate
limits during the first download.

## Profiles

| Profile | Cache | Context limit (input+output) | Loader margin | Notes |
|---|---|---:|---:|---|
| `q8_384k` | Q8, 393,216 tokens | 392,960 | 128 MiB | lm_head reserved on GPU 0; tightest layout |
| `q8_320k` | Q8, 327,680 tokens | 327,424 | 256 MiB | |
| `q8_fast` | Q8, 262,144 tokens | 261,888 | 192 MiB | |
| `fp16` | FP16, 196,608 tokens | 196,352 | 256 MiB | stock speed-first profile |
| `mtp` | same as `fp16` | 196,352 | 256 MiB | stock MTP d2 instead of DFlash2 |

`-gs` is ExLlamaV3's **per-component load budget**, not a VRAM split. For the
MTP profile the effective split is `59,63` because the MTP head is loaded onto
GPU 0 first; see `glm/glm_profile.py` for the validated values.

## API

- Base URL: `http://127.0.0.1:8012/v1` (model id `GLM-5.3-Flash`, no API key)
- `POST /v1/chat/completions`, `POST /v1/completions`, SSE streaming
- Reasoning is returned as `reasoning_content`, the answer as `content`
- OpenAI `tools` / `tool_choice` / `role: tool` messages are supported
- `GET /health` reports cache format, context, speculation mode, draft ring size
- `GET /v1/models` reports the active context limit
- Defaults: `reasoning_effort=high`, `temperature=1.0`, `top_p=0.95`,
  `max_tokens=32768`. One request runs at a time; the rest wait up to 120 s.
- `timings` exposes ExLlamaV3-native `prompt_ms` / `predicted_ms` plus draft
  accepted/rejected counts, so PP/TG/acceptance can be measured from the client
  without trusting SSE arrival times.

## How the memory budget works

The 384K Q8 profile is the largest layout that fits on 2× 64 GB with this
checkpoint:

- Target weights: ~116.6 GiB, fully resident across both GPUs
- Q8 KV cache (11 MLA layers, latent quantized): ~4.51 GiB at 384K
- DFlash2 drafter: ~0.96 GiB weights + 8K-token GPU ring cache for its KV
- lm_head reserved on GPU 0 before the transformer split (`--q8-head-gpu0`)
- Autosplit margin 128 MiB (`EXL3_AUTOSPLIT_MARGIN_MB`)

448K and 512K fail during the final transformer-layer load with this
checkpoint and layout; they are not offered as profiles.

### Why the drafter KV is a ring cache

The drafter uses a fixed 2048-token sliding window. Its logical cache still
matches the target context, but the physical GPU storage is an 8,192-token
ring that recycles pages; absolute RoPE positions and the attention window are
unchanged. This is what lets the drafter serve a 384K request without a
384K-token draft cache. The ring is verified for equivalence against the
unbounded cache (bit-exact outputs over repeated wraps); see the experiment
notes in the blog post this repo accompanies.

## DFlash2 drafter conversion

`dflash2/convert_drafter.py` wraps the stock ExLlamaV3 converter with two
process-local adjustments (the ExLlamaV3 tree itself is never modified):

1. `k_proj`/`v_proj` stay BF16 — serving builds a fused context-KV weight from
   the qkv projection's K/V rows, which must remain a plain tensor.
2. `DFlash2DynConv.kernel_projection` Linears are given a qmap so they enter
   the quantization budget (env `DRAFT_QUANT_KERNEL_PROJ=0` to keep them BF16).

`dflash2/package_native.py` then restores every unquantized tensor byte-for-byte
from the BF16 source and writes native `quantization_config` / `tensor_storage`
metadata so ExLlamaV3 1.5.4 loads the result directly. Result: 36 quantized
linears at 6-bit trellis, ~0.96 GiB drafter weights.

Conversion uses synthetic-Hessian calibration only (DFlash2's
`uncalibrated_quantize` capability); no target-model forwards are involved.

## Engine notes

- `k_hcfuse` fuses the four mHC decode launches per sublayer site into two,
  bit-exact against the stock kernels (shadow-check mode: `GLM53_K_HCFUSE_CHECK=1000`).
  It JIT-compiles on first decode (SM80, nvcc); subsequent runs reuse the build cache.
- The async generator runs native GPU steps on a dedicated worker thread so the
  HTTP loop stays responsive; disconnect cancellation is shielded until the
  in-flight GPU step and native job cleanup finish.
- Q8 staging bounds packed-pool gather copies to 32K-token tiles and pins the
  quantized-append workspace to two fixed buffers per GPU, so repeated requests
  do not accumulate VRAM.
- Prompt KV reuse is disabled in the DFlash2 mode: the drafter ring has no
  snapshot of previous requests, so every request prefills from scratch. Warm
  prefill throughput and the reuse-time saving are separate metrics.

## Known limitations

- One active request; concurrent requests are queued (up to 120 s wait).
- Multimodal input and JSON-schema-forced output are unsupported.
- Q8 cache changes numerics; no bit-exactness vs FP16 is claimed for outputs.
- 448K/512K do not fit with this checkpoint; the goal of reaching 1M context
  remains open.
- The conversion pipeline and profiles were validated on this specific host
  (Ryzen 5 5600X, DDR4, PCIe Gen2 x8). Other hosts may need different margins.

## Credits

- [ExLlamaV3](https://github.com/turboderp/exllamav3) — turboderp (engine, MIT)
- [GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) — turboderp (3.05bpw checkpoint)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) — incoai (BF16 drafter)
- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) — conversion reference
- [0xSero/glm53-flash-offload](https://github.com/0xSero/glm53-flash-offload) — k_hcfuse kernels
- GLM-5.3-Flash — Z.ai

This repo does not redistribute any model weights; everything is downloaded
from the original Hugging Face repos at their pinned revisions.

## License

MIT — see [LICENSE](LICENSE).
