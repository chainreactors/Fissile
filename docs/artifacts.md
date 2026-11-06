# Configuration and artifacts

Fissile loads external artifacts; none are stored in this repository.

| Artifact | Contents | Integrity check |
|---|---|---|
| Merged model | BF16 checkpoint directory | Effective weight hash per projection |
| Projection packs | One safetensors file per projection: `q` (packed E2M1), `s` (E4M3 bytes), `g` (FP32 global scale) | File hash in the manifest |
| Manifest | Effective-weight hash, packed-file hash, shape and path for every projection | — |
| TensorFold library | `fissile build-tensorfold` output | Build receipt (`.build.json`) |
| FP4 GEMM library | `fissile build-fp4-gemm` output (`gb10-mixed-v2`) | Build receipt (`.build.json`) |
| SiLU table | `scripts/build_silu_table.py` output | Configured SHA256 |

## Prepare packs

`fissile quantize` runs native BF16 forwards of a local model over calibration
inputs and prepares every projection pack on the GPU. It needs Torch,
Transformers, Triton and safetensors (the `torch` extra).

```bash
fissile quantize --model /path/to/merged-model --calibration calibration.jsonl \
  --packs-dir artifacts/packs --output artifacts/manifest.json \
  [--profile qwen3.5-dense-9b] [--recipe mixed-fp4-fp8-w4a16-v1]
```

| Step | Method |
|---|---|
| Calibration inputs | JSONL, one tokenized input per line (`input_ids`); other fields are ignored. Use representative, unlabeled inputs, separate from evaluation data |
| Input moments | Projections sharing an input share one moment. Each forward contributes at most 256 evenly spaced rows: FP32 Gram products, FP64 accumulation. The output head sees the final position only |
| Block scales | Each 16-value block takes the E4M3 scale with the lowest error under its 16×16 input moment |
| GPTQ | Columns in descending input-moment order, 1% damping, FP64 inverse |
| Output | `<packs-dir>/<projection>.safetensors` and a manifest with effective-weight and pack hashes |

For the 9B profile on one GB10, 96 calibration inputs (37,230 tokens) take 15
minutes including model load, with 72 GiB peak GPU memory (model plus FP64
input moments); the packs score like the published ones
([measurements](../bench/README.md#pack-preparation)).

`fissile.quantization.prepare.prepare_projection(weight, hessian, output)`
prepares one projection from its full input moment.

Packs prepared elsewhere:

```bash
# Validate existing packs and write a manifest
fissile import-packs --prepared /path/to/prepared-packs --output artifacts/manifest.json \
  [--profile qwen3.5-dense-9b] [--recipe mixed-fp4-fp8-w4a16-v1]
# Copy validated packs into a portable directory
fissile export-bundle --manifest artifacts/manifest.json --artifacts-dir /path/to/packs --output /path/to/bundle
```

## Configuration file

```json
{
  "format": "fissile-config-v2",
  "profile": "qwen3.5-dense-9b",
  "recipe": "mixed-fp4-fp8-w4a16-v1",
  "kernel_provider": "gb10-mixed-v2",
  "manifest": "../artifacts/manifest.json",
  "artifacts_dir": "../artifacts/packs",
  "report_dir": "../artifacts/sglang-run-001",
  "resources": {
    "tensorfold_library": {"path": "../artifacts/libfissile_tensorfold.so"},
    "silu_lut": {"path": "../artifacts/silu.safetensors", "sha256": "<sha256>"},
    "fp4_gemm_library": {"path": "../artifacts/libfissile_fp4_gemm.so"}
  }
}
```

Paths resolve relative to the configuration file. `resources` must match the
selected kernel provider: `gb10-mixed-v1` and `sm120-mixed-v1` take
`tensorfold_library` and `silu_lut`; `gb10-mixed-v2` also takes
`fp4_gemm_library`. Each SGLang process writes installation and coverage
reports to a new `report_dir`. `fissile validate --config config.json` checks
every artifact. `fissile-config-v1` files ([example](../examples/config.json)) are
also accepted and select `gb10-mixed-v1`.

## Build native libraries

```bash
fissile build-tensorfold --output artifacts/libfissile_tensorfold.so --cuda-arch sm_121
fissile build-fp4-gemm --output artifacts/libfissile_fp4_gemm.so \
  --cutlass-dir "$(python -c 'import flashinfer,os;print(os.path.dirname(flashinfer.__file__))')/data/cutlass"
python scripts/build_silu_table.py --output artifacts/silu.safetensors
```

Library builds are CPU-only. `build-fp4-gemm` accepts any CUTLASS 4.5 source
tree, such as the copy bundled with FlashInfer. The SiLU table build runs on the
GPU and prints its SHA256.

## SGLang environment

Python 3.12, Torch 2.13.0+cu130, SGLang 0.5.21, Transformers 5.12.1,
Triton 3.7.1 and CuTe DSL 4.8.0. Install SGLang in its own environment, then
place CuTe DSL 4.8.0 ahead of SGLang's dependencies on `PYTHONPATH`:

```bash
FISSILE_DSL_DIR="$HOME/.local/share/fissile/cute-4.8"
python -m pip install --target "$FISSILE_DSL_DIR" 'nvidia-cutlass-dsl[cu13]==4.8.0'
export PYTHONPATH="$FISSILE_DSL_DIR/nvidia_cutlass_dsl/dsl_packages:$FISSILE_DSL_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import importlib.metadata as m; print(m.version("nvidia-cutlass-dsl"))'  # 4.8.0
python -m pip install .
```

Ninja and a matching CUDA toolchain are required for SGLang's JIT kernels.

## vLLM environment

Python 3.12, Torch 2.13.0+cu130, vLLM 0.29.0, Triton 3.7.1 and CuTe DSL 4.8.0.
Install vLLM in its own environment and place CuTe DSL 4.8.0 ahead of vLLM's
pinned 4.6.2 on `PYTHONPATH`, as for SGLang. `ninja` must be on `PATH`: vLLM
compiles FlashInfer sampling kernels on first start.

| Environment variable | Default | Accepted values |
|---|---|---|
| `FISSILE_VLLM_KV_CACHE_BYTES` | 2 GiB | Integer between 256 MiB and 1 TiB |

## SGLang resource settings

| Environment variable | Default | Accepted values |
|---|---|---|
| `FISSILE_SGLANG_MEM_FRACTION_STATIC` | `0.35` | Number strictly between 0 and 1 |
| `FISSILE_SGLANG_NCCL_PORT` | Automatic | Integer 1024–65535 |

## GB10 memory

On GB10, CUDA reports `MemFree` as free memory, so clean page cache from
recently read checkpoints counts as used. Check `MemFree` before starting an
engine; dropping cached pages of your own files with
`posix_fadvise(POSIX_FADV_DONTNEED)` makes that memory available to CUDA.
`fissile quantize` drops the checkpoint's cached pages after loading the model.
