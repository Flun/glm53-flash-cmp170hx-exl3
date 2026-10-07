"""Opt-in, previous-request prefix reuse for the single-sequence DFlash ring.

Target KV pages and recurrent checkpoints remain owned by EXL3. Only the draft's
2048-token SWA history is copied to host RAM at matching native checkpoints.
No engine files or running processes are modified by importing this module.
"""
from dataclasses import dataclass


PAGE_SIZE = 256
WINDOW_TOKENS = 2048


@dataclass(frozen=True)
class DraftCheckpoint:
    position: int
    key: bytes
    prefix: object
    page_indices: tuple
    tensors: tuple

    @property
    def host_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.prefix, *self.tensors))


class PreviousRequestCache:
    """Keep the prompt-end checkpoint plus the latest decode checkpoint.

    Requests run serially. A miss invalidates the previous request entirely;
    this deliberately does not implement a multi-session cache. Missing target
    pages/checkpoints, changed tokens and cancellations all fall back to cold PP.
    """
    def __init__(self, generator):
        self.generator = generator
        cache = generator.draft_cache
        ring = getattr(cache, "dflash_ring_tokens", 0)
        if (not generator.dflash_draft or generator.max_batch_size != 1
                or generator.recurrent_cache is None or generator.model.loaded_tp
                or ring < WINDOW_TOKENS + generator.max_chunk_size + 8
                or ring % PAGE_SIZE):
            raise ValueError("Prefix reuse requires a single-sequence recurrent target and DFlash GPU ring")
        self.ring_pages = ring // PAGE_SIZE
        self.storage = tuple(cache.get_all_tensors())
        if not self.storage or any(t is None or t.shape[0] != self.ring_pages
                                   or t.shape[1] != PAGE_SIZE for t in self.storage):
            raise ValueError("Unsupported draft ring tensor layout")
        self.published = ()
        self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
        self.job = None
        self.hits = self.misses = self.restored_tokens = 0
        self.last_reason = "empty"

    def invalidate(self, reason):
        self.published = ()
        self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
        self.job = None
        self.last_reason = reason

    def _cold(self):
        g = self.generator
        g.pagetable.reset_page_table()
        g.recurrent_cache.prune_stranded()

    def begin(self, job):
        import torch
        # The ring patch already rejects parallel jobs. Refuse to reuse states
        # for features whose rewind/requeue semantics this integration doesn't cover.
        supported = (len(job.sequences) == 1 and not job.is_requeued
                     and not job.embeddings and not job.filters and not job.banned_strings
                     and job.prefix_token is None and job.orig_max_rq_tokens is None)
        seq = job.sequences[0]
        ids = seq.sequence_ids.torch()
        candidates = self.published
        self.published = ()
        self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
        self.job = job if supported else None
        if supported:
            for cp in sorted(candidates, key=lambda cp: cp.position, reverse=True):
                stash = self.generator.recurrent_cache.get(cp.key)
                if (cp.position < ids.shape[-1] and stash is not None
                        and stash["position"] == cp.position
                        and self.generator.pagetable.is_resumable(cp.key)
                        and torch.equal(ids[:, :cp.position], cp.prefix)):
                    self.plan = cp
                    # Keep the restored boundary available even if no new native
                    # checkpoint is reached in this very short follow-up.
                    self.prompt_checkpoint = cp
                    self.last_reason = "matched"
                    break
        if self.plan is None:
            self.misses += 1
            self.last_reason = "prefix_or_state_miss" if supported else "unsupported_request"
            self._cold()

    def restrict(self, job):
        if job is not self.job:
            return
        # Cap native KV/recurrent reuse to the boundary for which draft state is
        # also available. Never let EXL3 restore a newer, unpaired checkpoint.
        pages = self.plan.position // PAGE_SIZE if self.plan else 0
        seq = job.sequences[0]
        seq.max_cached_pages = pages
        job.all_unique_hashes = list(set(seq.page_hashes[:pages]))

    def restore(self, job, allocate):
        if job is not self.job or self.plan is None:
            return
        cp = self.plan
        seq = job.sequences[0]
        if (seq.kv_position != cp.position or job.recurrent_state is None
                or job.recurrent_state.position != cp.position
                or job.cached_pages * PAGE_SIZE != cp.position):
            # KV eviction/allocation must not silently resume from an older
            # target checkpoint with a newer draft ring. Reallocate cold instead.
            job.deallocate_pages()
            self._cold()
            self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
            seq.kv_position = 0
            seq.prefill_complete = False
            seq.max_cached_pages = 0
            job.all_unique_hashes = []
            job.cached_pages = job.cached_tokens = job.total_pages = job.non_sequential_pages = 0
            job.last_recurrent_checkpoint_pos = None
            self.misses += 1
            self.last_reason = "allocation_miss"
            allocate(job)
            return
        for dst, saved in zip(self.storage, cp.tensors):
            for row, index in enumerate(cp.page_indices):
                dst[index].copy_(saved[row])
        self.hits += 1
        self.restored_tokens += cp.position
        self.last_reason = "restored"
        self.plan = None  # The prompt slot retains it; don't keep a third host snapshot alive.

    def capture(self, job):
        import torch
        if job is not self.job:
            return
        seq = job.sequences[0]
        pos = seq.kv_position
        if not pos or pos % PAGE_SIZE or job.recurrent_state.position != pos:
            return
        key = seq.allocated_pages[pos // PAGE_SIZE - 1].phash
        stash = self.generator.recurrent_cache.get(key)
        if stash is None or stash["position"] != pos:
            return
        end_page = pos // PAGE_SIZE
        indices = tuple(p % self.ring_pages
                        for p in range(max(0, end_page - WINDOW_TOKENS // PAGE_SIZE), end_page))
        # Copy raw packed Q8 values AND scales (or raw FP16 K/V), without
        # dequantization or a transient GPU clone. Blocking host copies ensure
        # the snapshot is complete before the ring can wrap/overwrite it.
        saved = []
        for src in self.storage:
            dst = torch.empty((len(indices), *src.shape[1:]), dtype=src.dtype, device="cpu")
            for row, index in enumerate(indices):
                dst[row].copy_(src[index])
            saved.append(dst)
        cp = DraftCheckpoint(pos, key, seq.sequence_ids.torch_slice(0, pos).clone(),
                             indices, tuple(saved))
        prompt_boundary = (len(seq.input_ids) - 1) // PAGE_SIZE * PAGE_SIZE
        if pos <= prompt_boundary:
            self.prompt_checkpoint = cp
        else:
            self.decode_checkpoint = cp

    def finish(self, results):
        for result in results:
            if result.get("job") is not self.job:
                continue
            if result.get("stage") == "error":
                self.invalidate("generation_error")
                return
            if result.get("eos"):
                self.published = tuple(cp for cp in (self.prompt_checkpoint, self.decode_checkpoint)
                                       if cp is not None)
                self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
                self.job = None
                return

    def stats(self):
        snapshots = self.published + tuple(cp for cp in
                                          (self.prompt_checkpoint, self.decode_checkpoint, self.plan)
                                          if cp is not None)
        unique = {id(cp): cp for cp in snapshots}
        return {"mode": "previous_request", "hits": self.hits, "misses": self.misses,
                "restored_tokens": self.restored_tokens, "host_bytes": sum(cp.host_bytes for cp in unique.values()),
                "snapshots": len(unique), "last_reason": self.last_reason}


def enable_prefix_cache(generator):
    """Install instance-gated hooks; GPU verification is required before adoption."""
    from exllamav3.generator.generator import Generator
    from exllamav3.generator.job import Job

    if not getattr(Job.prepare_for_queue, "_glm_dflash_prefix", False):
        prepare, allocate, stash = Job.prepare_for_queue, Job.allocate_pages, Job.maybe_stash_recurrent
        iterate, cancel = Generator.iterate, Generator.cancel

        def prepare_hook(self, g, *args, **kwargs):
            result = prepare(self, g, *args, **kwargs)
            manager = getattr(g, "_glm_dflash_prefix_cache", None)
            if manager:
                manager.restrict(self)
            return result

        def allocate_hook(self):
            result = allocate(self)
            manager = getattr(self.generator, "_glm_dflash_prefix_cache", None)
            if manager:
                manager.restore(self, allocate)
            return result

        def stash_hook(self, cache, *args, **kwargs):
            before = self.last_recurrent_checkpoint_pos
            result = stash(self, cache, *args, **kwargs)
            manager = getattr(self.generator, "_glm_dflash_prefix_cache", None)
            if manager and self.last_recurrent_checkpoint_pos != before:
                manager.capture(self)
            return result

        def iterate_hook(self):
            manager = getattr(self, "_glm_dflash_prefix_cache", None)
            try:
                result = iterate(self)
            except BaseException:
                if manager:
                    manager.invalidate("engine_error")
                raise
            if manager:
                manager.finish(result)
            return result

        def cancel_hook(self, job):
            manager = getattr(self, "_glm_dflash_prefix_cache", None)
            if manager and (job in self.active_jobs or job in self.pending_jobs):
                manager.invalidate("cancelled")
            return cancel(self, job)

        prepare_hook._glm_dflash_prefix = True
        Job.prepare_for_queue, Job.allocate_pages, Job.maybe_stash_recurrent = prepare_hook, allocate_hook, stash_hook
        Generator.iterate, Generator.cancel = iterate_hook, cancel_hook

    generator._glm_dflash_prefix_cache = PreviousRequestCache(generator)
