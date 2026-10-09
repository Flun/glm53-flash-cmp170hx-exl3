#!/usr/bin/env python3
"""OpenAI-compatible text API for the existing GLM EXL3/MTP installation.

One request runs at a time to keep the validated two-GPU memory configuration.
The model's own Jinja chat template is used without requiring transformers.
"""
import argparse
import asyncio
import anyio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Literal
import uuid

# Make sibling modules importable when launched as a script from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel, ConfigDict, Field
from glm_profile import CONTEXT_RESERVE, MODEL_CONTEXT_LIMIT, MODEL_ID, PORT, GENERATION_DEFAULTS
from glm_tools import prepare_tools, ToolSplitter
from glm_paths import KERNELS_DIR, HCFUSE_ENV, HCFUSE_CHECK_ENV

runtime = {}


# The deployed ExLlamaV3 generator page size is fixed at 256 tokens.
PREFILL_PAGE_SIZE = 256


def prefill_chunk_size(args):
    """Validate before model allocation; optionally cap serving only, not autosplit."""
    load_chunk = args.chunk_size
    cap = getattr(args, "prefill_chunk_size", None)
    for name, value in (("--chunk_size", load_chunk), ("--prefill-chunk-size", cap)):
        if value is None and name == "--prefill-chunk-size":
            continue
        if type(value) is not int or value <= 0 or value % PREFILL_PAGE_SIZE:
            raise ValueError(f"{name} must be a positive multiple of {PREFILL_PAGE_SIZE}")
    if cap is not None and cap > load_chunk:
        raise ValueError("--prefill-chunk-size must not exceed --chunk_size")
    return load_chunk if cap is None else cap



class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolDefinition(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str = Field(min_length=1)
    arguments: str | dict = "{}"


class ToolCall(BaseModel):
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: FunctionCall


class NamedToolChoice(BaseModel):
    type: Literal["function"]
    function: dict[str, str]


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict] | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_ID
    messages: list[Message] | None = None
    prompt: str | None = None
    max_tokens: int = Field(default=GENERATION_DEFAULTS["max_tokens"], ge=1, le=MODEL_CONTEXT_LIMIT)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=MODEL_CONTEXT_LIMIT)
    temperature: float = Field(default=GENERATION_DEFAULTS["temperature"], ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=GENERATION_DEFAULTS["top_p"], gt=0, le=1, allow_inf_nan=False)
    frequency_penalty: float = Field(default=0, ge=-2, le=2, allow_inf_nan=False)
    presence_penalty: float = Field(default=0, ge=-2, le=2, allow_inf_nan=False)
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    stream_options: dict | None = None
    ignore_eos: bool = Field(default=False, strict=True)
    min_tokens: int = Field(default=0, ge=0, le=MODEL_CONTEXT_LIMIT)
    # DSH/pi-ai sends store=false for OpenAI chat requests. This API does not
    # persist completions; reject an explicit storage request before generation.
    store: bool | None = Field(default=False, strict=True)
    reasoning_effort: Literal["low", "high", "max"] = GENERATION_DEFAULTS["reasoning_effort"]
    n: Literal[1] = 1
    user: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | None = None
    parallel_tool_calls: bool = True


def render_prompt(template, messages, effort, tools=None):
    for message in messages:
        if isinstance(message["content"], list):
            if any(part.get("type") != "text" or not isinstance(part.get("text"), str)
                   for part in message["content"]):
                raise ValueError("Only text message content is supported")
    return template.render(messages=messages, tools=tools, add_generation_prompt=True,
                           reasoning_effort=effort)


class ReasoningSplitter:
    """Separate template-prefilled <think> output, even across token boundaries."""
    def __init__(self):
        self.thinking = True
        self.pending = ""

    def feed(self, text, final=False):
        if not self.thinking:
            return {"content": text} if text else {}
        self.pending += text
        marker = "</think>"
        if marker in self.pending:
            thought, content = self.pending.split(marker, 1)
            self.pending = ""
            self.thinking = False
            return {key: value for key, value in
                    (("reasoning_content", thought), ("content", content)) if value}
        held = 0
        if not final:
            for count in range(1, len(marker)):
                if self.pending.endswith(marker[:count]):
                    held = count
        emit = self.pending[:-held] if held else self.pending
        self.pending = self.pending[-held:] if held else ""
        return {"reasoning_content": emit} if emit else {}


@asynccontextmanager
async def lifespan(app):
    from exllamav3 import AsyncGenerator, GreedySampler, model_init
    args = runtime["args"]
    serving_chunk_size = prefill_chunk_size(args)
    if getattr(args, "q8_staging", False):
        from glm_q8_staging import install
        install()
    if getattr(args, "q8_head_gpu0", False):
        from glm_q8_layout import install
        install(reserve_head=bool(args.draft_model_dir) and not args.mtp)
    if getattr(args, "fp16_draft_cache", False):
        from glm_q8_layout import keep_mtp_cache_fp16
        keep_mtp_cache_fp16()
    dflash = bool(args.draft_model_dir) and not args.mtp
    if dflash:
        from glm_dflash import bounded_draft_cache, pin_drafter
        bounded_draft_cache()
        pin_drafter(0 if getattr(args, "q8_head_gpu0", False) else 1)
    method = "DFlash2 EXL3 6bpw K7" if dflash else "MTP d2"
    print(f"Loading GLM: GPU 0+1, full VRAM, {method} + k_hcfuse", flush=True)
    model, config, cache, tokenizer, draft_model, _, draft_cache = model_init.init(
        args, quiet=True, progress=False)
    sys.path.insert(0, str(KERNELS_DIR))
    import k_hcfuse
    k_hcfuse.install(model)
    if dflash:
        if not draft_model.caps.get("dflash_draft") or draft_model.config.block_size != 8:
            raise ValueError("This API requires a DFlash2 block-size-8 drafter")
        from glm_dflash import preserve_fused_exports
        preserve_fused_exports(model)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, **kwargs)
    template = env.from_string((Path(args.model_dir) / "chat_template.jinja").read_text())
    runtime.update(tokenizer=tokenizer, config=config, template=template,
                   lock=asyncio.Lock(), loaded_at=int(time.time()),
                   context_length=min(config.max_position_embeddings,
                                      args.cache_size - CONTEXT_RESERVE))
    from glm_async import responsive_generator_class
    runtime["generator"] = responsive_generator_class(AsyncGenerator)(
        model=model, cache=cache, tokenizer=tokenizer, sampler=GreedySampler(),
        draft_model=draft_model, draft_cache=draft_cache,
        num_draft_tokens=args.num_draft_tokens, max_batch_size=1,
        max_chunk_size=serving_chunk_size, recurrent_checkpoint_interval=2048)
    if dflash and getattr(args, "dflash_prefix_cache", False):
        from glm_dflash_prefix import enable_prefix_cache
        enable_prefix_cache(runtime["generator"].generator)
        print("GLM DFlash2: previous-request prefix cache enabled (paired host snapshots)", flush=True)
    print(f"GLM ready: http://0.0.0.0:{args.port}/v1 · context {runtime['context_length']} "
          f"· cache {args.cache_size}", flush=True)
    try:
        yield
    finally:
        await runtime["generator"].close()


app = FastAPI(title="GLM-5.3-Flash EXL3 API", version="1.0", lifespan=lifespan,
              description="Text chat/completions, SSE streaming, separate reasoning_content. "
                          "Context length is reported by /health and /v1/models; "
                          "one active request; defaults: high reasoning, temperature 1, "
                          "top_p 0.95, max_tokens 32768 including reasoning. "
                          "Function calls are returned to the client for execution. "
                          "Multimodal input and JSON schema output are unsupported.")


@app.exception_handler(HTTPException)
async def http_error(request, error):
    return JSONResponse(status_code=error.status_code, content={"error": {
        "message": str(error.detail), "type": "invalid_request_error", "code": error.status_code}})


@app.exception_handler(RequestValidationError)
async def validation_error(request, error):
    message = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return JSONResponse(status_code=400, content={"error": {
        "message": message, "type": "invalid_request_error", "code": 400}})


@app.get("/", include_in_schema=False)
async def home():
    return RedirectResponse("/docs")


@app.get("/health")
async def health():
    if runtime["generator"].error:
        raise HTTPException(503, "Generation engine failed; restart the GLM service")
    dflash = (bool(getattr(runtime["args"], "draft_model_dir", None))
              and not getattr(runtime["args"], "mtp", False))
    generator = getattr(runtime["generator"], "generator", runtime["generator"])
    draft_cache = getattr(generator, "draft_cache", None)
    prefix_cache = getattr(generator, "_glm_dflash_prefix_cache", None)
    return {"status": "ok", "engine": "ExLlamaV3", "model": MODEL_ID, "context_length": runtime["context_length"],
            "cache_tokens": runtime["args"].cache_size,
            "max_chunk_size": generator.max_chunk_size,
            "load_chunk_size": runtime["args"].chunk_size,
            "prefill_chunk_cap": getattr(runtime["args"], "prefill_chunk_size", None),
            "recurrent_checkpoint_interval": generator.recurrent_checkpoint_interval,
            "cache_format": cache_format(),
            "q8_staging": getattr(runtime["args"], "q8_staging", False),
            "draft_cache_format": ("FP16" if getattr(runtime["args"], "fp16_draft_cache", False)
                                   else cache_format()),
            "output_gpu": 0 if getattr(runtime["args"], "q8_head_gpu0", False) else 1,
            "workspace_margin_mb": int(os.environ.get("EXL3_AUTOSPLIT_MARGIN_MB", "256")),
            "mtp_depth": 0 if dflash else 2,
            "speculative_method": "dflash2" if dflash else "mtp",
            "draft_num_tokens": getattr(runtime["args"], "num_draft_tokens", None) or 2,
            "draft_ring_tokens": getattr(draft_cache, "dflash_ring_tokens", None),
            "prompt_cache_reuse": not dflash or prefix_cache is not None,
            "prompt_cache": prefix_cache.stats() if prefix_cache is not None else None,
            "busy": runtime["lock"].locked(),
            "generation_defaults": GENERATION_DEFAULTS}


class TokenizeRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str | None = None
    content: str | None = None


@app.post("/tokenize")
async def tokenize(body: TokenizeRequest):
    if body.model not in (MODEL_ID, "glm", Path(runtime["args"].model_dir).name):
        raise HTTPException(404, "Unknown model")
    text = body.prompt if body.prompt is not None else body.content
    if text is None:
        raise HTTPException(400, "prompt or content is required")
    return {"token_length": runtime["tokenizer"].encode(text, encode_special_tokens=True).shape[-1]}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model",
            "created": runtime["loaded_at"], "owned_by": "local",
            "context_length": runtime["context_length"], "cache_format": cache_format()}]}


def cache_format():
    quant = getattr(runtime["args"], "cache_quant", None)
    return f"Q{quant}" if quant is not None else "FP16"


def stream_text(result, tokenizer):
    text = result.get("text", "")
    # EXL3's piece join can emit an interior replacement character when the next
    # token completes a Korean UTF-8 character and also contains printable text.
    # Decode the held token group together, keeping stop-string trimming intact.
    if ("\ufffd" in text and result.get("token_ids") is not None
            and result.get("eos_reason") != "stop_string"):
        decoded = tokenizer.decode(result["token_ids"], decode_special_tokens=True)
        return decoded[0] if isinstance(decoded, list) else decoded
    return text


async def complete(body, request, chat):
    from exllamav3 import AsyncJob, ComboSampler, GreedySampler
    if body.store:
        raise HTTPException(400, "Stored completions are unsupported; use store=false")
    if body.model not in (MODEL_ID, "glm", Path(runtime["args"].model_dir).name):
        raise HTTPException(404, f"Unknown model: {body.model}")
    tools = []
    tool_choice = (body.tool_choice.model_dump() if isinstance(body.tool_choice, NamedToolChoice)
                   else body.tool_choice)
    if chat:
        if not body.messages or body.prompt is not None:
            raise HTTPException(400, "A nonempty messages list is required")
        try:
            messages, tools = prepare_tools(
                [m.model_dump(exclude_none=True) for m in body.messages],
                [t.model_dump(exclude_none=True) for t in body.tools or []],
                tool_choice, body.parallel_tool_calls)
            prompt = render_prompt(runtime["template"],
                                   messages, body.reasoning_effort, tools)
        except (ValueError, KeyError) as error:
            raise HTTPException(400, str(error)) from error
    else:
        if body.tools or tool_choice not in (None, "none", "auto"):
            raise HTTPException(400, "Function tools require /v1/chat/completions")
        if not body.prompt or body.messages is not None:
            raise HTTPException(400, "A nonempty string prompt is required")
        prompt = body.prompt
    ids = runtime["tokenizer"].encode(prompt, encode_special_tokens=True)
    prompt_tokens = ids.shape[-1]
    limit = body.max_completion_tokens or body.max_tokens
    if body.min_tokens > limit:
        raise HTTPException(400, "min_tokens exceeds the output limit")
    context_length = runtime["context_length"]
    if prompt_tokens + limit > context_length:
        raise HTTPException(400, f"Prompt ({prompt_tokens}) + output ({limit}) exceeds context {context_length}")
    stops = list(runtime["config"].eos_token_id_list)
    if not body.ignore_eos and not body.min_tokens:
        stops += ["<|user|>", "<|observation|>"]
    stops += [body.stop] if isinstance(body.stop, str) else (body.stop or [])
    if any(not stop for stop in stops):
        raise HTTPException(400, "Stop strings must not be empty")
    sampler = (GreedySampler() if body.temperature == 0 and body.frequency_penalty == 0
               and body.presence_penalty == 0 else ComboSampler(
                   temperature=body.temperature, top_p=body.top_p,
                   freq_p=body.frequency_penalty, pres_p=body.presence_penalty))
    lock = runtime["lock"]
    try:
        await asyncio.wait_for(lock.acquire(), timeout=120)
    except TimeoutError:
        raise HTTPException(429, "Server busy; retry later")
    try:
        if runtime["generator"].error:
            raise HTTPException(503, "Generation engine failed; restart the service")
        job = AsyncJob(runtime["generator"], input_ids=ids, max_new_tokens=limit,
                       min_new_tokens=limit if body.ignore_eos else body.min_tokens,
                       stop_conditions=stops, sampler=sampler, seed=body.seed,
                       decode_special_tokens=True)
    except Exception:
        lock.release()
        raise
    ident = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
    created = int(time.time())
    splitter = ReasoningSplitter() if chat else None
    tool_splitter = ToolSplitter(tools, body.parallel_tool_calls) if tools else None
    collected = {"content": "", "reasoning_content": ""}
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens}
    reason = "stop"
    timings = {}

    def envelope(choices, streaming=False):
        return {"id": ident, "object": ("chat.completion.chunk" if streaming else "chat.completion")
                if chat else "text_completion", "created": created, "model": MODEL_ID, "choices": choices}

    def event(payload):
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    async def results():
        nonlocal reason
        try:
            if body.stream and chat:
                yield envelope([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}], True)
            async for result in job:
                if await request.is_disconnected():
                    return
                if result["stage"] != "streaming":
                    continue
                text = stream_text(result, runtime["tokenizer"])
                delta = splitter.feed(text, final=result["eos"]) if chat else {"content": text}
                deltas = []
                if "reasoning_content" in delta:
                    deltas.append({"reasoning_content": delta["reasoning_content"]})
                if tool_splitter:
                    deltas.extend(tool_splitter.feed(delta.get("content", ""), final=result["eos"],
                                  truncated=result.get("eos_reason") == "max_new_tokens"))
                elif "content" in delta:
                    deltas.append({"content": delta["content"]})
                for delta in deltas:
                    for key in ("content", "reasoning_content"):
                        collected[key] += delta.get(key, "")
                    if body.stream:
                        choice = {"index": 0, "finish_reason": None}
                        choice.update({"delta": delta} if chat else {"text": delta.get("content", "")})
                        yield envelope([choice], True)
                if result["eos"]:
                    reason = "length" if result.get("eos_reason") == "max_new_tokens" else "stop"
                    usage["completion_tokens"] = int(result["new_tokens"])
                    usage["total_tokens"] = prompt_tokens + usage["completion_tokens"]
                    for native, public in [("time_prefill", "prompt_ms"),
                                           ("time_generate", "predicted_ms"),
                                           ("time_enqueued", "queue_ms")]:
                        if result.get(native) is not None:
                            timings[public] = result[native] * 1000
                    cached_tokens = min(prompt_tokens, max(0, int(result.get("cached_tokens", 0))))
                    usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
                    timings.update(prompt_n=prompt_tokens, predicted_n=usage["completion_tokens"],
                                   source="exllamav3", cached_tokens=cached_tokens, cache_n=cached_tokens)
                    if result.get("accepted_draft_tokens") is not None:
                        accepted = result["accepted_draft_tokens"]
                        drafted = accepted + result.get("rejected_draft_tokens", 0)
                        usage.update(draft_tokens=drafted, draft_tokens_accepted=accepted)
                        timings.update(draft_n=drafted, draft_n_accepted=accepted)
            if tool_splitter and tool_splitter.calls and reason != "length":
                reason = "tool_calls"
            if reason != "length" and (tool_choice == "required" or isinstance(tool_choice, dict)):
                if not tool_splitter or not tool_splitter.calls:
                    raise ValueError("Model did not produce the requested function call")
            if body.stream:
                choice = {"index": 0, "finish_reason": reason}
                choice.update({"delta": {}} if chat else {"text": ""})
                yield {**envelope([choice], True), "timings": timings}
                if (body.stream_options or {}).get("include_usage"):
                    yield {**envelope([], True), "usage": usage}
        finally:
            try:
                # StreamingResponse cancels its AnyIO scope on disconnect.
                # Keep the request lock until the current GPU step finishes and
                # the native job/page table is safely removed.
                with anyio.CancelScope(shield=True):
                    await job.cancel()
            finally:
                lock.release()

    if body.stream:
        async def stream():
            try:
                async for payload in results():
                    yield event(payload)
                yield "data: [DONE]\n\n"
            except Exception as error:
                print(f"Generation failed: {error!r}", flush=True)
                traceback.print_exception(error)
                oom = "out of memory" in str(error).lower()
                yield event({"error": {"message": "GPU memory exhausted; see service log" if oom
                                       else "Generation failed; see service log",
                                       "type": "server_error", "code": "out_of_memory" if oom else "generation_failed"}})
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    try:
        async for _ in results():
            pass
    except Exception as error:
        print(f"Generation failed: {error!r}", flush=True)
        traceback.print_exception(error)
        raise HTTPException(500, "Generation failed; see service log") from error
    choice = {"index": 0, "finish_reason": reason}
    choice.update({"message": {"role": "assistant", **collected}} if chat
                  else {"text": collected["content"], "logprobs": None})
    if chat and tool_splitter and tool_splitter.calls:
        choice["message"]["tool_calls"] = tool_splitter.calls
    return {**envelope([choice]), "usage": usage, "timings": timings}


@app.post("/v1/chat/completions")
async def chat_completions(body: CompletionRequest, request: Request):
    return await complete(body, request, chat=True)


@app.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request):
    return await complete(body, request, chat=False)


if __name__ == "__main__":
    from exllamav3 import model_init
    import uvicorn
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache=True, add_draft_model_args=True, default_chunk_size=2048)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--prefill-chunk-size", type=int,
                        default=os.environ.get("GLM53_PREFILL_CHUNK_SIZE"),
                        help="Cap serving prefill chunks independently of loader/autosplit --chunk_size "
                             "(default: GLM53_PREFILL_CHUNK_SIZE when set)")
    parser.add_argument("--q8-staging", action="store_true",
                        help="Bound packed-pool staging copies for the Q8 profile")
    parser.add_argument("--q8-head-gpu0", action="store_true",
                        help="Keep the output projection resident on GPU 0")
    parser.add_argument("--fp16-draft-cache", action="store_true",
                        help="Keep the MTP draft layer cache FP16")
    parser.add_argument("--dflash-prefix-cache", action="store_true",
                        help="Opt in to previous-request DFlash prefix reuse with paired host snapshots")
    args = parser.parse_args()
    try:
        prefill_chunk_size(args)
    except ValueError as error:
        parser.error(str(error))
    if not ((args.mtp and not args.draft_model_dir and args.num_draft_tokens == 2)
            or (not args.mtp and args.draft_model_dir and args.num_draft_tokens == 7)):
        parser.error("Use either --mtp -ndt 2 or -dm <DFlash2 EXL3 6bpw> -ndt 7")
    if (args.q8_staging or args.q8_head_gpu0 or args.fp16_draft_cache) and args.cache_quant != "8":
        parser.error("Q8 memory options require -cq 8")
    if args.dflash_prefix_cache and args.mtp:
        parser.error("--dflash-prefix-cache requires DFlash2; MTP already uses native prefix caching")
    if args.q8_staging and args.autosplit_max_batch_size != 1:
        parser.error("--q8-staging requires --autosplit_max_batch_size 1")
    runtime["args"] = args
    # k_hcfuse gates itself on this flag; glm_api always calls install() explicitly.
    os.environ.setdefault(HCFUSE_ENV, "1")
    uvicorn.run(app, host="0.0.0.0", port=args.port, timeout_graceful_shutdown=5)
