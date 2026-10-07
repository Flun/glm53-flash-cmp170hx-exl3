"""Local DFlash2 integration for stock ExLlamaV3 1.5.4 and k_hcfuse."""


def pin_drafter(device=1):
    """Keep the small drafter beside the target head, avoiding vocab-logit copies."""
    from exllamav3.model.model import Model
    import torch

    original = Model.load
    if getattr(original, "_glm_dflash_device", False):
        return

    def load(self, *args, **kwargs):
        if self.caps.get("dflash_draft"):
            kwargs.pop("use_per_device", None)
            kwargs.pop("reserve_per_device", None)
            kwargs["device"] = torch.device(f"cuda:{device}")
        return original(self, *args, **kwargs)

    load._glm_dflash_device = True
    Model.load = load


def preserve_fused_exports(model):
    """Flush deferred mHC applies at taps and export the stock collapsed state.

    k_hcfuse normally falls back for every block if any state tap is requested.
    During decode, finish only each tapped block's deferred apply and reproduce
    TransformerBlock's mean -> half -> clamp export. Prefill keeps stock exports.
    Install immediately after k_hcfuse.install; no engine files are changed.
    """
    import torch
    import k_hcfuse
    from exllamav3.modules.transformer import TransformerBlock

    original = TransformerBlock.forward
    if getattr(original, "_glm_dflash_exports", False):
        return
    blocks = {id(m) for m in model.modules if isinstance(m, TransformerBlock)
              and m.layer_scalar_f is None and k_hcfuse._block_ok(m)}

    def forward(self, x, params, out_dtype=None):
        taps = params.get("export_state_layers")
        eligible = (id(self) in blocks and taps and not params.get("prefill")
                    and "capture" not in params and "quant_preserve" not in params
                    and x.dim() == 4 and x.dtype == torch.float and x.is_contiguous()
                    and x.shape[0] * x.shape[1] <= k_hcfuse.MAXR
                    and out_dtype in (None, torch.float)
                    and self.out_dtype in (None, torch.float))
        if not eligible:
            return original(self, x, params, out_dtype)
        params.pop("export_state_layers")
        try:
            result = original(self, x, params, out_dtype)
        finally:
            params["export_state_layers"] = taps
        if self.layer_idx in taps and params.get("layer_instance", 0) == 0:
            k_hcfuse.flush()
            state = result.mean(dim=2).half().clamp_(-65504.0, 65504.0)
            params.setdefault("export_states", []).append(state)
        return result

    forward._glm_dflash_exports = True
    TransformerBlock.forward = forward
    print("GLM DFlash2: k_hcfuse preserved with stock state exports", flush=True)


def bounded_draft_cache(tokens=8192):
    """Use a GPU ring for the fixed SWA drafter, keeping absolute RoPE positions.

    The ring exceeds window (2048) + prefill chunk (2048) + draft block (8).
    Logical cache size still matches the target; only drafter physical storage is
    bounded. One active sequence is required. By default prompts run cold. The
    opt-in prefix manager pairs target checkpoints with snapshots of this ring.
    """
    import inspect
    import torch
    from exllamav3.cache import Cache, CacheLayer_fp16, CacheLayer_quant
    from exllamav3.architecture.dflash2 import DFlash2Model
    from exllamav3.generator.generator import Generator

    original = Cache.__init__
    if getattr(original, "_glm_dflash_ring", False):
        return
    if tokens < 2048 + 2048 + 8 or tokens % 256:
        raise ValueError("DFlash ring must cover window + prefill chunk + block")

    class RingFP16(CacheLayer_fp16):
        def __init__(self, config, attention, cache_id, max_num_tokens, **kw):
            super().__init__(config, attention, cache_id, min(tokens, max_num_tokens), **kw)

    class RingQ8(CacheLayer_quant):
        def __init__(self, config, attention, cache_id, max_num_tokens, **kw):
            super().__init__(config, attention, cache_id, min(tokens, max_num_tokens), **kw)

    signature = inspect.signature(original)

    def init(self, *args, **kw):
        bound = signature.bind(self, *args, **kw)
        model = bound.arguments["model"]
        if isinstance(model, DFlash2Model):
            # EXL3 stores the number of *past* keys, excluding the query itself.
            if any(a.sliding_window != 2047 for a in model.attn_modules):
                raise ValueError("Bounded DFlash cache requires all 2048-token SWA layers")
            layer = bound.arguments.get("layer_type") or CacheLayer_fp16
            bound.arguments["layer_type"] = RingQ8 if layer is CacheLayer_quant else RingFP16
            bound.arguments["max_batch_size"] = 1
            self.dflash_ring_tokens = min(tokens, bound.arguments["max_num_tokens"])
        return original(*bound.args, **bound.kwargs)

    init._glm_dflash_ring = True
    Cache.__init__ = init
    tables = {}

    def ring_params(params, cache):
        if not getattr(cache, "dflash_ring_tokens", None):
            return params
        table = params["block_table"]
        if table.shape[0] != 1:
            raise ValueError("DFlash ring supports one active sequence")
        key = (table.shape[-1], cache.dflash_ring_tokens // 256)
        if key not in tables:
            ring = (torch.arange(key[0], dtype=torch.int32) % key[1]).unsqueeze(0)
            ring._static_dev_cache = True
            tables[key] = ring
        return {**params, "block_table": tables[key]}

    original_forward = DFlash2Model.forward
    original_update = DFlash2Model.update_kv_from_target

    def forward(self, input_ids, params, **kw):
        updated = ring_params(params, params["cache"])
        table = params["block_table"]
        params["block_table"] = updated["block_table"]
        try:
            return original_forward(self, input_ids=input_ids, params=params, **kw)
        finally:
            params["block_table"] = table

    def update(self, target_hidden, cache, params, lengths=None):
        return original_update(self, target_hidden=target_hidden, cache=cache,
                               params=ring_params(params, cache), lengths=lengths)

    DFlash2Model.forward = forward
    DFlash2Model.update_kv_from_target = update
    original_enqueue = Generator.enqueue

    def enqueue(self, job):
        manager = None
        if self.dflash_draft and getattr(self.draft_cache, "dflash_ring_tokens", None):
            if self.num_remaining_jobs() or self.max_batch_size != 1 or isinstance(job, list):
                raise ValueError("DFlash ring requires serialized single-sequence jobs")
            self.enable_defrag = False
            manager = getattr(self, "_glm_dflash_prefix_cache", None)
            if manager is not None:
                manager.begin(job)
            else:
                self.pagetable.reset_page_table()
                if self.recurrent_cache is not None:
                    self.recurrent_cache.prune_stranded()
        try:
            return original_enqueue(self, job)
        except Exception:
            if manager is not None:
                manager.invalidate("enqueue_error")
            raise

    Generator.enqueue = enqueue
    print(f"GLM DFlash2: {tokens}-token GPU draft ring; cold prompt prefill", flush=True)
