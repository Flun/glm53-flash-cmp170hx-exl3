"""Bound temporary packed-pool gathers without changing EXL3 cache arithmetic.

The FP16 attention staging pool remains on the GPU. Only the copies of packed
pages/scales are bounded to 32K tokens; the stock CUDA dequantizer is retained.
Install before model_init.init so autosplit measures the same implementation.
"""
STAGE_TOKENS = 32768


def stage_packed_pool(pool_c, qc, pool_r, block_table, page_size, pool_len, D_c, D_r):
    import torch
    from exllamav3.ext import exllamav3_ext as ext

    pool_s, bits = qc
    groups = D_c // 32
    num_pages = (pool_len + page_size - 1) // page_size
    pages = block_table[0, :num_pages].long()
    packed = pool_c.reshape(-1, page_size, groups * bits)
    scales = pool_s.reshape(-1, page_size, groups)
    rows = num_pages * page_size
    out = torch.empty((rows, D_c), dtype=torch.half, device=pool_c.device)
    rope = (torch.empty((num_pages, page_size, D_r), dtype=pool_r.dtype,
                        device=pool_r.device) if D_r else pool_r)
    pages_per_tile = max(1, STAGE_TOKENS // page_size)
    for first in range(0, num_pages, pages_per_tile):
        last = min(first + pages_per_tile, num_pages)
        selected = pages[first:last]
        pc = packed[selected]
        ps = scales[selected]
        ext.dequant_cache_cont(pc.view(-1, groups * bits), ps.view(-1, groups),
                               out[first * page_size:last * page_size], 0.0)
        if D_r:
            rope[first:last].copy_(pool_r.reshape(-1, page_size, D_r)[selected])
        # Release before allocating the next tile, avoiding two overlapping gathers.
        del pc, ps
    identity = torch.arange(num_pages, dtype=torch.int32, device=pool_c.device).unsqueeze(0)
    return out.view(num_pages, page_size, D_c), rope, identity


def install():
    from exllamav3.modules.attention_fn import dsa_triton
    from exllamav3.modules.attention_fn import bc_attn, bc_mla
    from exllamav3.modules import mla_attn
    dsa_triton._stage_packed_pool = stage_packed_pool
    # This server serializes requests and always uses batch size 1. Keep all
    # supported query lengths, but avoid reserving unused batch-8 decode statics.
    bc_attn.MAX_BSZ = bc_mla.MAX_BSZ = mla_attn._bc_max_bsz = 1
    bc_attn.MAX_R = bc_attn.MAX_QLEN
    # Quant append otherwise retains another workspace for every distinct tail
    # length. Two fixed buffers per GPU bound this across repeated API requests.
    from exllamav3.util.tensor import g_tensor_cache
    original = g_tensor_cache.get
    if not getattr(original, "_glm_bounded_append", False):
        def get(device, shape, dtype, x=""):
            if x in ("mla_qtmp", "mla_stmp") and len(shape) == 2 and 0 < shape[0] <= 2048:
                return original(device, (2048, shape[1]), dtype, "glm_" + x)[:shape[0]]
            return original(device, shape, dtype, x)
        get._glm_bounded_append = True
        g_tensor_cache.get = get
