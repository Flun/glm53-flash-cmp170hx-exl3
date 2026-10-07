# GLM-5.3-Flash EXL3 서빙 — 2× 64GB GPU (CMP 170HX / SM80)

P2P 미지원 64GB GPU 두 장(CMP 170HX 등)에서 **GLM-5.3-Flash**를
**DFlash2 K7 추측 디코딩**으로 풀 VRAM 서빙하는 스택입니다. 대안으로 스톡
**MTP d2**도 지원합니다. ExLlamaV3 소스는 수정하지 않고, 이 서버가 실행되는
동안에만 켜지는 프로세스 로컬 몽키패치로 엔진을 조정합니다.

[English README](README.md) · 이 문서(한국어)

## 포함 내용

| 구성 요소 | 경로 | 설명 |
|---|---|---|
| OpenAI 호환 API 서버 | `glm/glm_api.py` | chat/completions + SSE, native PP/TG 시간, 도구 호출 |
| DFlash2 통합 패치 | `glm/glm_dflash.py` | drafter 배치 고정, 8K GPU 링 캐시, fused export 보존 |
| 반응형 비동기 제너레이터 | `glm/glm_async.py` | GPU 스텝을 HTTP 이벤트 루프 밖으로, 취소 보호 |
| Q8 메모리 배치 패치 | `glm/glm_q8_layout.py`, `glm/glm_q8_staging.py` | lm_head GPU 0 배치, staging 제한, batch-1 디코드 작업공간 |
| GLM 도구 호출 변환 | `glm/glm_tools.py` | GLM 네이티브 XML 도구 호출 → OpenAI `tool_calls` |
| mHC fused 디코드 커널 | `kernels/k_hcfuse.py` | 0xSero k_hcfuse, ExLlamaV3 1.5.4 호환 |
| DFlash2 BF16 → EXL3 6bpw 변환 | `dflash2/convert_drafter.py`, `dflash2/package_native.py` | 36개 선형층 6bit, drafter 약 0.96 GiB |
| 이전 요청 프리픽스 재사용 | `glm/glm_dflash_prefix.py` | 반복 프롬프트 캐시 히트, `serve.sh` 기본 켜짐 |
| 설치/서빙 런처 | `scripts/setup.sh`, `scripts/serve.sh` | 첫 실행에 Hugging Face에서 자동 다운로드 |

미포함(범위 밖): 실험 캠페인의 벤치마크 하네스와 원시 측정 데이터, 그리고 이
서버가 원래 들어 있던 대시보드 통합 코드.

## 검증된 구성

| 항목 | 값 |
|---|---|
| Target 모델 | `turboderp/GLM-5.3-Flash-exl3` @ `3.05bpw` (revision 고정) |
| 엔진 | ExLlamaV3 1.5.4 (SM80 네이티브 빌드) |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2` BF16 → 직접 변환 EXL3 6bpw (약 0.96 GiB) |
| 추측 디코딩 | DFlash2 K7 (기본) 또는 MTP d2 |
| KV 캐시 | Q8 (MLA latent), FP16 프로필도 있음 |
| 서비스 컨텍스트 | 384K (Q8) / 320K / 256K / 192K (FP16) |
| 최대 요청 | 입력+출력 약 392,960 토큰 (384K 프로필) |
| GPU | 2× 64GB, SM80, P2P 미지원 (host bounce D2D) |
| 분할 예산 | `-gs 62,63` (각 숫자는 컴포넌트별 로딩 예산, VRAM 분할이 아님) |

모든 가중치는 HBM에 상주합니다: CPU expert tier 없음, SSD paging 없음,
실행 중 weight 전송 없음. Strata식 HBM-first 방식을 ExLlamaV3 위에 재구성한
것이고, 컨텍스트에 비례해 늘어나는 캐시·drafter 쪽을 줄이는 방향입니다.

## 빠른 시작

```bash
# 1. 환경 (venv + SM80용 ExLlamaV3 1.5.4 빌드 + API 의존성)
scripts/setup.sh

# 2. 서빙. 첫 실행에 target(약 125GB)과 BF16 drafter(약 2GB)를 Hugging Face에서
#    받고, drafter를 EXL3 6bpw로 변환합니다(1회, 수십 분). 이후 실행은 ./models 재사용.
scripts/serve.sh --profile q8_384k

# 3. 사용
curl http://127.0.0.1:8012/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"안녕하세요"}],"max_tokens":512}'
```

필요 조건: `nvcc`가 있는 CUDA 툴킷(compute_80), Linux, 모델+변환 작업공간용
여유 디스크 약 140GB, 두 GPU 모두 다른 작업 없이 비어 있어야 합니다. GPU 여유
메모리가 약 64GB 미만이면 서버가 시작을 거부합니다.

공개 레포라 `HF_TOKEN`은 필수가 아니지만 첫 다운로드에서 rate limit에 걸리면
설정하세요.

## 프로필

| 프로필 | 캐시 | 컨텍스트 상한(입력+출력) | 로더 여유 | 비고 |
|---|---|---:|---:|---|
| `q8_384k` | Q8, 393,216 토큰 | 392,960 | 128 MiB | lm_head GPU 0 예약, 가장 빡빡한 배치 |
| `q8_320k` | Q8, 327,680 토큰 | 327,424 | 256 MiB | |
| `q8_fast` | Q8, 262,144 토큰 | 261,888 | 192 MiB | |
| `fp16` | FP16, 196,608 토큰 | 196,352 | 256 MiB | 기존 속도 우선 프로필 |
| `mtp` | `fp16`과 동일 | 196,352 | 256 MiB | DFlash2 대신 스톡 MTP d2 |

`-gs`는 ExLlamaV3의 **컴포넌트별 로딩 예산**이지 VRAM 분할이 아닙니다. MTP
프로필의 실질 분할은 `59,63`입니다(MTP head를 GPU 0에 먼저 올림). 검증값은
`glm/glm_profile.py`에 있습니다.

## API

- Base URL: `http://127.0.0.1:8012/v1` (모델 id `GLM-5.3-Flash`, API 키 불필요)
- `POST /v1/chat/completions`, `POST /v1/completions`, SSE 스트리밍
- 생각 과정은 `reasoning_content`, 답변은 `content`
- OpenAI `tools` / `tool_choice` / `role: tool` 메시지 지원
- `GET /health`가 캐시 정밀도, 컨텍스트, 추측 모드, draft 링 크기와 프리픽스 캐시
  통계(`prompt_cache`: hits/misses/복원 토큰)를 보고
- `GET /v1/models`가 실행 중 컨텍스트 상한을 보고
- 기본값: `reasoning_effort=high`, `temperature=1.0`, `top_p=0.95`,
  `max_tokens=32768`. 요청은 하나씩 실행되고 나머지는 최대 120초 대기.
- `timings`에 ExLlamaV3 native `prompt_ms` / `predicted_ms`와 draft
  accepted/rejected 수, `cached_tokens`(`usage.prompt_tokens_details.cached_tokens`
  에도 포함)를 노출해서, SSE 도착 시각 대신 실제 생성 시간 기준으로
  PP/TG/수락률과 캐시 히트를 클라이언트에서 측정할 수 있습니다.

## 메모리 예산의 동작 방식

384K Q8 프로필은 이 체크포인트 기준 2× 64GB에 들어가는 가장 큰 배치입니다:

- Target 가중치: 약 116.6 GiB, 두 GPU에 전부 상주
- Q8 KV 캐시(11개 MLA 레이어, latent 양자화): 384K에서 약 4.51 GiB
- DFlash2 drafter: 가중치 약 0.96 GiB + KV용 8K 토큰 GPU 링 캐시
- lm_head를 transformer 분할 전에 GPU 0에 예약 (`--q8-head-gpu0`)
- Autosplit 여유 128 MiB (`EXL3_AUTOSPLIT_MARGIN_MB`)

448K와 512K는 이 체크포인트·배치에서 마지막 transformer 레이어 로딩 중
VRAM 부족으로 실패하므로 프로필로 제공하지 않습니다.

### drafter KV를 링 캐시로 쓰는 이유

drafter는 고정 2048-token 슬라이딩 윈도우를 씁니다. 논리 캐시 크기는 target
컨텍스트와 동일하게 유지되지만, 실제 GPU 저장소는 8,192 토큰 링으로 페이지를
재활용합니다. 절대 RoPE 위치와 attention 윈도우는 그대로입니다. 이 덕분에
drafter가 384K 요청을 처리하면서도 384K짜리 draft 캐시를 따로 잡지 않습니다.
링은 무한 캐시와의 동등성(반복 순환 후 bit-exact 출력) 검증을 통과했습니다.
자세한 내용은 이 레포와 함께 공개한 실험 글을 참고하세요.

## DFlash2 drafter 변환

`dflash2/convert_drafter.py`는 스톡 ExLlamaV3 컨버터를 두 가지 프로세스 로컬
조정으로 감쌉니다(ExLlamaV3 트리 자체는 수정하지 않음):

1. `k_proj`/`v_proj`는 BF16 유지 — 서빙이 qkv projection의 K/V 행으로 fused
   context-KV 가중치를 만들기 때문에 일반 텐서여야 합니다.
2. `DFlash2DynConv.kernel_projection` Linear에 qmap을 부여해 양자화 예산에
   넣습니다(BF16 유지는 env `DRAFT_QUANT_KERNEL_PROJ=0`).

`dflash2/package_native.py`는 미양자화 텐서를 BF16 원본에서 바이트 단위로
복원하고 native `quantization_config` / `tensor_storage` 메타데이터를 써서
ExLlamaV3 1.5.4가 바로 로드하게 합니다. 결과: 6bit trellis 36개 선형층,
drafter 가중치 약 0.96 GiB.

변환은 synthetic-Hessian 캘리브레이션만 사용합니다(DFlash2의
`uncalibrated_quantize` capability). target 모델 forward는 개입하지 않습니다.

## 엔진 노트

- `k_hcfuse`는 서브레이어 사이트당 mHC 디코드 런치 4개를 2개로 합치고,
  스톡 커널과 비트정합입니다(섀도 체크: `GLM53_K_HCFUSE_CHECK=1000`).
  첫 디코드에서 JIT 컴파일(SM80, nvcc)되고 이후 실행은 빌드 캐시를 재사용합니다.
- 비동기 제너레이터는 native GPU 스텝을 전용 워커 스레드에서 실행해 HTTP 루프를
  막지 않고, 연결 끊김 취소는 진행 중 GPU 스텝과 native job 정리가 끝날 때까지
  보호됩니다.
- Q8 staging은 packed-pool gather 복사를 32K 토큰 타일로 제한하고 양자화 append
  작업공간을 GPU당 두 개 고정 버퍼로 묶어, 반복 요청에서 VRAM이 누적되지 않게 합니다.
- DFlash2 모드의 프롬프트 KV 재사용은 옵트인이며 `scripts/serve.sh`로 시작하면
  기본 켜짐입니다(`--dflash-prefix-cache`; 끄려면 `GLM53_PREFIX_CACHE=0`).
  drafter 링에는 이전 요청의 스냅샷이 없으므로, 프리픽스 매니저가 target
  recurrent checkpoint마다 최근 2048 drafter 토큰의 호스트 RAM 스냅샷을 짝지어
  보관하고 캐시 히트에서 둘 다 복원합니다. 프리픽스 미스, checkpoint 불일치,
  취소된 요청, 엔진 오류는 모두 cold 프리필로 폴백합니다.

  검증 호스트 실측(temperature 0, seed 42, 1K/8K/32K 반복): 반복 1K 프롬프트의
  첫 토큰 대기 1.50초 → 0.65초, 8K 대화 후속 질문 7.86초 → 1.11초(토큰 93.98%
  재사용), 32K 후속 질문 28.49초 → 1.12초(98.42% 재사용). 호스트 스냅샷은
  checkpoint당 CPU RAM 약 43MiB를 쓰고, GPU VRAM 사용량은 측정 노이즈 범위에서
  변화 없음. 반복 입력의 greedy 출력은 cold 실행과 일치했지만, 8K/32K 실제
  텍스트 후속 요청은 native GPU 비교에서 전체 재처리 기준 출력과 일부 달랐습니다.
  결함 위치는 문서화만 하고 수정하지 않았으므로, 재사용 경로는 측정된 반복
  케이스에서 검증된 것으로 보고 일반적인 무손실은 주장하지 않습니다.
- warm 프리필 처리량과 재사용으로 얻는 시간 절약은 별개 지표입니다.

## 알려진 제한

- 동시 요청 1개만 실행, 나머지는 대기(최대 120초).
- 멀티모달 입력과 JSON schema 강제 출력은 미지원.
- Q8 캐시는 수치를 바꾸며, FP16 대비 출력 bit-exact를 주장하지 않습니다.
- 이 체크포인트로는 448K/512K가 안 들어가며, 1M 컨텍스트는 여전히 목표로 남아 있습니다.
- 변환 파이프라인과 프로필은 이 특정 호스트(Ryzen 5 5600X, DDR4, PCIe Gen2 x8)에서
  검증했습니다. 다른 호스트에서는 여유 값을 조정해야 할 수 있습니다.

## 크레딧

- [ExLlamaV3](https://github.com/turboderp/exllamav3) — turboderp (엔진, MIT)
- [GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) — turboderp (3.05bpw 체크포인트)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) — incoai (BF16 drafter)
- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) — 변환 참고
- [0xSero/glm53-flash-offload](https://github.com/0xSero/glm53-flash-offload) — k_hcfuse 커널
- GLM-5.3-Flash — Z.ai

이 레포는 모델 가중치를 재배포하지 않으며, 모든 파일은 원본 Hugging Face 레포에서
고정 revision으로 다운로드됩니다.

## 라이선스

MIT — [LICENSE](LICENSE) 참고.
