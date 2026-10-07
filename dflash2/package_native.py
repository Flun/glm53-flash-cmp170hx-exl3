"""MiaAI packaging rule adapted to native ExLlamaV3 1.5.4 metadata.

Keep 6-bit EXL3 parts; restore every unquantized tensor byte-for-byte from BF16.
Unlike the vLLM package script, keep native quantization_config/tensor_storage.

Usage:
  python package_native.py -i <BF16 DFlash2 dir> -o <EXL3 6bpw output dir> \
      [--reference-commit <sha>]

Run after convert_drafter.py; the converter writes the 6-bit trellis tensors
into <out>/model.safetensors and this script merges the unquantized BF16
tensors back in and finalizes the native metadata.
"""
import argparse
import json
import pathlib
import subprocess
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav3.conversion.quant_config import create_quantization_config_json

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True,
                    help="BF16 DFlash2 source directory (incoai/GLM-5.3-Flash-DFlash2)")
parser.add_argument("-o", "--output", required=True,
                    help="EXL3 6bpw output directory (same one convert_drafter.py wrote)")
parser.add_argument("--reference-commit", default=None,
                    help="Commit of the reference conversion repo, recorded in the summary")
args = parser.parse_args()
src = pathlib.Path(args.input)
out = pathlib.Path(args.output)

with safe_open(out / 'model.safetensors', framework='pt', device='cpu') as f:
    tensors = {k: f.get_tensor(k) for k in f.keys()}
quant = {k[:-8] for k in tensors if k.endswith('.trellis')}
restored = []
n_params = 0
with safe_open(src / 'model.safetensors', framework='pt', device='cpu') as f:
    for key in f.keys():
        t = f.get_tensor(key)
        n_params += t.numel()
        if key.endswith('.weight') and key[:-7] in quant:
            continue
        tensors[key] = t
        restored.append(key)
assert all(tensors[b + '.trellis'].shape[-1] == 96 for b in quant)
assert len(quant) == 36, len(quant)
tmp = out / 'model.pack.tmp'
save_file(tensors, tmp)
tmp.replace(out / 'model.safetensors')
cfg = json.loads((out / 'config.json').read_text())
cfg['tie_word_embeddings'] = False
cfg['quantization_config'].update(head_bits=16,
                                  calibration={'method': 'uncalibrated synthetic Hessian; no target forwards'},
                                  scope='dflash2_draft')
(out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
create_quantization_config_json(str(out))
size = sum(f.stat().st_size for f in out.glob('*.safetensors'))
summary = {
    'source': 'incoai/GLM-5.3-Flash-DFlash2',
    'source_revision': 'bf582e4eacc1810f76656d1811693ff6c6737d2a',
    'reference': 'https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks',
    'reference_commit': args.reference_commit,
    'converter': 'ExLlamaV3 1.5.4',
    'quantized_linears': len(quant),
    'trellis_bits': 6,
    'weights_bytes': size,
    'weights_GiB': size / 2**30,
    'source_parameters': n_params,
    'effective_overall_bpw': size * 8 / n_params,
    'bf16_restored_tensors': restored,
    'quantized_tensors': sorted(quant),
}
(out / 'draft_quant_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))
