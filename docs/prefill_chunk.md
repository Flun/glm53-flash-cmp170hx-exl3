# Prefill chunk controls

The loader/autosplit argument `--chunk_size` now also determines the serving generator’s `max_chunk_size`. Previously the generator always used 2048, even when the loader warmed and packed the model for a smaller value. Default 2048 behavior is unchanged.

Use `--prefill-chunk-size` or `GLM53_PREFILL_CHUNK_SIZE` to cap serving chunks independently of loader placement. The CLI value takes precedence over the environment. Both values must be positive integer multiples of the native 256-token page size; the optional serving cap must not exceed the loader chunk. Validation occurs before model allocation and again at lifespan entry. Recurrent checkpoints retain their 2048-token interval.

`/health` reports actual generator `max_chunk_size`, configured `load_chunk_size`, optional `prefill_chunk_cap` and checkpoint interval. A cap can reduce transient prefill demand, but increases launch overhead and may slow long prompts. It does not add active context capacity or change model weights.

The repo already includes the qualified responsive async generator and previous-request paired-prefix helper at base revision 5933112. Their code is unchanged by this patch. The existing repeat-case validation and broader losslessness limitation in README still apply. Sparse attention stage controls and native engine patches are separate changes and are not applied here.

Run `python -m unittest discover -s tests -p test_prefill_chunk.py -v` for CPU-only validation. The tests execute the actual API CLI/lifespan/health AST with fake loaders and generators; they do not import the native extension, initialize CUDA or load model weights.
