"""CPU-safe administration commands; no implicit engine launch."""

import argparse, json
from . import __version__


def main():
    p = argparse.ArgumentParser(prog="fissile")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser(
        "quantize",
        help="Calibrate a BF16 model and prepare every projection pack and a manifest (GPU)",
    )
    q.add_argument("--model", required=True, help="Local BF16 model directory")
    q.add_argument("--calibration", required=True, help="JSONL, one tokenized input (input_ids) per line")
    q.add_argument("--packs-dir", required=True, help="New directory for the projection packs")
    q.add_argument("--output", required=True, help="New manifest path")
    q.add_argument("--profile", help="Model profile (default: qwen3.5-dense-9b)")
    q.add_argument("--recipe", help="Precision recipe (default: mixed-fp4-fp8-w4a16-v1)")
    q.add_argument("--max-rows-per-forward", type=int, default=256, help="Sampled input rows per forward (1-256)")
    v = sub.add_parser(
        "validate", help="Verify every external packed weight and configuration"
    )
    v.add_argument("--config", required=True)
    b = sub.add_parser("build-tensorfold", help="CPU compilation; no GPU kernels")
    b.add_argument("--output", required=True)
    b.add_argument("--nvcc")
    b.add_argument("--cuda-arch", choices=("sm_120", "sm_121"), default="sm_121")
    g = sub.add_parser(
        "build-fp4-gemm",
        help="CPU compilation of the pinned SM121 CUTLASS NVFP4 GEMM (gb10-mixed-v2)",
    )
    g.add_argument("--output", required=True)
    g.add_argument("--cutlass-dir", required=True)
    g.add_argument("--nvcc")
    e = sub.add_parser(
        "export-bundle",
        help="Copy the manifest's projection packs into a portable directory",
    )
    e.add_argument("--manifest", required=True)
    e.add_argument("--artifacts-dir", required=True)
    e.add_argument("--output", required=True)
    sub.add_parser("backends")
    a = p.parse_args()
    if a.command == "quantize":
        import sys
        from .models import DEFAULT_PROFILE
        from .recipes import DEFAULT_RECIPE
        from .backends.quantize import quantize

        def progress(stage, done, total):
            if done == total or done % 8 == 0:
                print(f"{stage} {done}/{total}", file=sys.stderr, flush=True)

        m = quantize(
            a.model,
            a.calibration,
            a.packs_dir,
            a.output,
            profile=a.profile or DEFAULT_PROFILE,
            recipe=a.recipe or DEFAULT_RECIPE,
            max_rows=a.max_rows_per_forward,
            progress=progress,
        )
        print(json.dumps({"projections": len(m["projections"]), "manifest": a.output, "packs_dir": a.packs_dir}))
    elif a.command == "validate":
        from .config import Config

        c = Config.load(a.config)
        print(
            json.dumps(
                {
                    "valid": True,
                    "profile": c.profile,
                    "projections": len(c.plan.profile.projections),
                    "recipe": c.recipe,
                    "kernel_provider": c.kernel_provider,
                    "gpu_launched": False,
                }
            )
        )
    elif a.command == "build-tensorfold":
        from .build import build_tensorfold

        print(
            json.dumps(
                build_tensorfold(a.output, a.nvcc, cuda_arch=a.cuda_arch), indent=2
            )
        )
    elif a.command == "build-fp4-gemm":
        from .build import build_fp4_gemm

        print(json.dumps(build_fp4_gemm(a.output, a.cutlass_dir, a.nvcc), indent=2))
    elif a.command == "export-bundle":
        from .artifacts import export_bundle

        print(export_bundle(a.manifest, a.artifacts_dir, a.output))
    elif a.command == "backends":
        from .backends.registry import describe_backends

        print(json.dumps(describe_backends(), indent=2))


if __name__ == "__main__":
    main()
