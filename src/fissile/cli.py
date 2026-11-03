"""CPU-safe administration commands; no implicit engine launch."""

import argparse, json
from . import __version__


def main():
    p = argparse.ArgumentParser(prog="fissile")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser(
        "import-legacy", help="Validate legacy GPTQ files and write a portable manifest"
    )
    i.add_argument("--prepared", required=True)
    i.add_argument("--output", required=True)
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
        help="Copy approved external weight files; not a Git publication",
    )
    e.add_argument("--manifest", required=True)
    e.add_argument("--artifacts-dir", required=True)
    e.add_argument("--output", required=True)
    sub.add_parser("backends")
    a = p.parse_args()
    if a.command == "import-legacy":
        from .artifacts import import_legacy

        m = import_legacy(a.prepared, a.output)
        print(json.dumps({"projections": len(m["projections"]), "manifest": a.output}))
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
