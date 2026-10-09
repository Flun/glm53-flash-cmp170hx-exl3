"""CPU startup/health regression tests of the isolated chunk API's actual AST.

Model allocation and generator objects are fakes. The real lifespan and CLI
bodies run; no GPU/native extension/model weights are imported or allocated.
"""
import argparse
import ast
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / 'glm/glm_api.py'
TREE = ast.parse(API.read_text())


def api_scope(runtime):
    selected = [n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in ('prefill_chunk_size', 'verification_generator_kwargs', 'lifespan', 'health')]
    selected = [ast.parse(ast.unparse(n)).body[0] for n in selected]
    for n in selected:
        n.decorator_list = []
    scope = dict(runtime=runtime, PREFILL_PAGE_SIZE=256, sys=sys, os=os, asyncio=asyncio,
                 Path=Path, time=time, json=json, KERNELS_DIR=ROOT, CONTEXT_RESERVE=256,
                 MODEL_ID='glm', GENERATION_DEFAULTS={}, cache_format=lambda: 'Q8')
    class TemplateEnvironment:
        def __init__(self, **kwargs): self.filters = {}
        def from_string(self, text): return text
    scope['ImmutableSandboxedEnvironment'] = TemplateEnvironment
    async def close(*args): runtime['closed'] = True
    scope['close_loaded_models'] = close
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(API), 'exec'), scope)
    return scope


def fixtures(chunk, cap=None):
    events = []
    directory = tempfile.TemporaryDirectory(prefix='glm-chunk-cpu-')
    (Path(directory.name) / 'chat_template.jinja').write_text('template')
    args = NS(chunk_size=chunk, prefill_chunk_size=cap, model_dir=directory.name,
              cache_size=393216, draft_model_dir=None, mtp=True, num_draft_tokens=2,
              dflash_prefix_cache=False, port=8013)
    model = NS(loaded_tp=True)
    config = NS(max_position_embeddings=393216)
    def init(received, **kwargs):
        events.append(('load', received.chunk_size))
        return model, config, object(), object(), None, None, None
    class AsyncGenerator:
        async def close(self): self.closed = True
        def __init__(self, **kwargs):
            self.closed = False
            events.append(('generator', kwargs['max_chunk_size'], kwargs['recurrent_checkpoint_interval']))
            self.error = None
            self.generator = NS(**kwargs, dflash_draft=False, mtp_draft=True)
    exllama = ModuleType('exllamav3')
    exllama.AsyncGenerator = AsyncGenerator
    exllama.GreedySampler = lambda: object()
    exllama.model_init = NS(init=init)
    fuse = ModuleType('k_hcfuse'); fuse.install = lambda model: events.append('fuse')
    asynchronous = ModuleType('glm_async'); asynchronous.responsive_generator_class = lambda cls: cls
    return directory, args, events, {'exllamav3':exllama, 'k_hcfuse':fuse, 'glm_async':asynchronous}


class PrefillChunkContracts(unittest.TestCase):
    def test_default_serving_chunk_matches_loader_for_256_512_2048(self):
        for chunk in (256, 512, 2048):
            with self.subTest(chunk=chunk):
                runtime, events, health = self.startup(chunk)
                self.assertEqual(events[0], ('load', chunk))
                self.assertIn(('generator', chunk, 2048), events)
                self.assertEqual(health['max_chunk_size'], chunk)
                self.assertEqual(health['load_chunk_size'], chunk)
                self.assertIsNone(health['prefill_chunk_cap'])
                self.assertEqual(health['recurrent_checkpoint_interval'], 2048)
                self.assertTrue(runtime['closed'])

    def test_runtime_cap_preserves_2048_loader_and_checkpoint_interval(self):
        for cap in (256, 512, 2048):
            with self.subTest(cap=cap):
                _, events, health = self.startup(2048, cap)
                self.assertEqual(events[0], ('load', 2048))
                self.assertIn(('generator', cap, 2048), events)
                self.assertEqual((health['max_chunk_size'], health['load_chunk_size'], health['prefill_chunk_cap']),
                                 (cap, 2048, cap))
                self.assertEqual(health['recurrent_checkpoint_interval'], 2048)

    def test_invalid_loader_is_rejected_before_loader_or_generator_calls(self):
        for chunk in (0, -256, 1, 255, 257, 511, 1.5, True, None):
            with self.subTest(chunk=chunk): self.rejected_startup(chunk, None)

    def test_invalid_or_oversized_cap_is_rejected_before_allocation(self):
        for cap in (0, -256, 1, 255, 257, 511, 1.5, True, 4096):
            with self.subTest(cap=cap): self.rejected_startup(2048, cap)

    def test_missing_optional_cap_defaults_to_loader(self):
        scope = api_scope({})
        self.assertEqual(scope['prefill_chunk_size'](NS(chunk_size=512)), 512)

    def test_actual_health_reports_generator_value_even_if_args_change(self):
        _, _, health = self.startup(2048, 512, change_actual=256)
        self.assertEqual(health['max_chunk_size'], 256)
        self.assertEqual(health['prefill_chunk_cap'], 512)
        self.assertEqual(health['load_chunk_size'], 2048)



    def test_cli_uses_new_optional_cap_and_rejects_before_uvicorn(self):
        for argv, expected in (([], None), (['--prefill-chunk-size','512'], 512)):
            runtime, served = self.cli(argv)
            self.assertEqual(runtime['args'].chunk_size, 2048)
            self.assertEqual(runtime['args'].prefill_chunk_size, expected)
            self.assertEqual(len(served), 1)
        for argv in (['--prefill-chunk-size','257'], ['--prefill-chunk-size','4096'], ['--chunk_size','0']):
            runtime, served = self.cli(argv, expect_failure=True)
            self.assertNotIn('args', runtime); self.assertEqual(served, [])


    def test_environment_cap_resolves_to_same_runtime_value_and_cli_takes_precedence(self):
        runtime, served = self.cli([], environment={'GLM53_PREFILL_CHUNK_SIZE':'512'})
        self.assertEqual(runtime['args'].prefill_chunk_size, 512)
        self.assertEqual(api_scope({})['prefill_chunk_size'](runtime['args']), 512)
        self.assertEqual(runtime['args'].chunk_size, 2048)
        self.assertEqual(len(served), 1)
        runtime, _ = self.cli(['--prefill-chunk-size','256'],
                              environment={'GLM53_PREFILL_CHUNK_SIZE':'512'})
        self.assertEqual(runtime['args'].prefill_chunk_size, 256)
        for value in ('', '0', '-256', '257', '4096', 'abc', '1.5'):
            with self.subTest(value=value):
                runtime, served = self.cli([], expect_failure=True,
                                          environment={'GLM53_PREFILL_CHUNK_SIZE':value})
                self.assertNotIn('args', runtime); self.assertEqual(served, [])


    def test_production_candidate_excludes_all_verification_features(self):
        text = API.read_text()
        for marker in ('GLM53_VERIFY_', 'glm_tp_logit_trace', 'verification_generator_kwargs',
                       'install_target_only_cold_enqueue', 'close_loaded_models'):
            self.assertNotIn(marker, text)

    def startup(self, chunk, cap=None, change_actual=None):
        directory, args, events, modules = fixtures(chunk, cap)
        runtime = {'args':args}; scope = api_scope(runtime)
        async def run():
            lifecycle = scope['lifespan'](None)
            await anext(lifecycle)
            if change_actual is not None: runtime['generator'].generator.max_chunk_size = change_actual
            result = await scope['health']()
            await lifecycle.aclose()
            runtime['closed'] = runtime['generator'].closed
            return result
        try:
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {}), redirect_stdout(io.StringIO()):
                result = asyncio.run(run())
            return runtime, events, result
        finally: directory.cleanup()

    def rejected_startup(self, chunk, cap):
        directory, args, events, modules = fixtures(chunk, cap)
        scope = api_scope({'args':args})
        async def run(): await anext(scope['lifespan'](None))
        try:
            with patch.dict(sys.modules, modules), self.assertRaises(ValueError): asyncio.run(run())
            self.assertEqual(events, [])
        finally: directory.cleanup()

    def cli(self, argv, expect_failure=False, environment=None):
        runtime = {}; scope = api_scope(runtime); served = []
        exllama = ModuleType('exllamav3')
        def add_args(parser, **kwargs):
            parser.add_argument('--chunk_size', type=int, default=kwargs['default_chunk_size'])
            for key,value in dict(mtp=True,draft_model_dir=None,num_draft_tokens=2,cache_quant='8',autosplit_max_batch_size=1).items():
                parser.set_defaults(**{key:value})
        exllama.model_init = NS(add_args=add_args)
        uvicorn = ModuleType('uvicorn'); uvicorn.run = lambda *args,**kwargs: served.append(kwargs)
        scope.update(argparse=argparse,PORT=8013,HCFUSE_ENV='GLM_CHUNK_CPU_TEST_HCFUSE',app=object())
        main = next(n for n in TREE.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
        code = compile(ast.Module(body=main.body,type_ignores=[]),str(API),'exec')
        with patch.dict(sys.modules, {'exllamav3':exllama,'uvicorn':uvicorn}), patch.object(sys,'argv',['glm_api_chunk.py']+argv), patch.dict(os.environ,{} if environment is None else environment,clear=True), redirect_stderr(io.StringIO()):
            if expect_failure:
                with self.assertRaises(SystemExit) as error: exec(code,scope)
                self.assertEqual(error.exception.code,2)
            else: exec(code,scope)
        return runtime,served


if __name__=='__main__': unittest.main(verbosity=2)
