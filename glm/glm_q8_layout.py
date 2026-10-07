"""Place only the final GLM output projection on GPU 0 to use its spare VRAM.

Transformer blocks keep their normal sequential split. The patch is scoped to
the GLM lm_head load; no expert weights or token caches leave the GPUs.
"""
import inspect


def keep_mtp_cache_fp16():
    """Use stock FP16 cache arithmetic for the single MTP draft layer."""
    from exllamav3.cache import Cache, CacheLayer_fp16, CacheLayer_quant

    original = Cache.__init__
    if getattr(original, "_glm_fp16_draft", False):
        return
    signature = inspect.signature(original)

    def init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        model = bound.arguments["model"]
        if (getattr(model, "component", None) == "mtp"
                and bound.arguments.get("layer_type") is CacheLayer_quant):
            bound.arguments["layer_type"] = CacheLayer_fp16
            print("GLM Q8: MTP draft cache kept FP16", flush=True)
        return original(*bound.args, **bound.kwargs)

    init._glm_fp16_draft = True
    Cache.__init__ = init


def install(reserve_head=False):
    from exllamav3.model.model_ls import Model_LSMixin
    import torch

    original = Model_LSMixin._load_autosplit
    if getattr(original, "_glm_head_gpu0", False):
        return
    signature = inspect.signature(original)

    def load(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        modules = bound.arguments["modules"]
        head = modules[-1]
        enabled = (head.key == "lm_head" and head.caps.get("logits_output")
                   and getattr(self, "component", "text") == "text"
                   and bound.arguments["active_devices"] == [0, 1])
        if not enabled:
            yield from original(self, *args, **kwargs)
            return
        load_head = head.load

        # DFlash is smaller than MTP and can allow autosplit to pack one more
        # transformer on GPU 0. Reserve the relocated head before planning that
        # split, so its weights are included in every physical headroom check.
        if reserve_head:
            load_head(torch.device("cuda:0"), max_chunk_size=bound.arguments["max_chunk_size"])
            print("GLM Q8: lm_head reserved on GPU 0 before transformer split", flush=True)

        def on_gpu0(device, *head_args, **head_kwargs):
            if reserve_head:
                return
            result = load_head(torch.device("cuda:0"), *head_args, **head_kwargs)
            print("GLM Q8: lm_head resident on GPU 0", flush=True)
            return result

        head.load = on_gpu0
        try:
            yield from original(self, *args, **kwargs)
            # The stock loader checks GPU 1 at this position. Also require spare
            # physical + reusable allocator memory on the output projection's GPU.
            free, _ = torch.cuda.mem_get_info(0)
            reusable = free + torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            minimum = 512 if reserve_head else 1024
            print(f"GLM Q8: output GPU reusable workspace {reusable / 2**20:.1f} MiB", flush=True)
            if reusable < minimum * 2**20:
                raise torch.cuda.OutOfMemoryError(
                    f"GLM output GPU has {reusable / 2**20:.1f} MiB workspace; requires {minimum} MiB")
        finally:
            head.load = load_head

    load._glm_head_gpu0 = True
    Model_LSMixin._load_autosplit = load
