"""Build only the vendored TensorFold C ABI wrapper; never launches GPU work."""

import json
from pathlib import Path
import os, shutil, subprocess
from .config import sha256


def build_tensorfold(output, nvcc=None, *, cuda_arch="sm_121"):
    if cuda_arch not in {"sm_120", "sm_121"}:
        raise ValueError("Supported CUDA targets are sm_120 and sm_121")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    compiler = nvcc or shutil.which("nvcc")
    if not compiler:
        raise RuntimeError("CUDA 13 toolchain nvcc is required on PATH")
    root = Path(__file__).parent
    src = root / "kernels/tensorfold.cu"
    vendor = root / "_vendor/tensorfold"
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(compiler),
        "-std=c++17",
        "-O3",
        "--fmad=false",
        "-DFISSILE_CUDA_MINOR=" + cuda_arch[-1],
        "-gencode=arch=compute_" + cuda_arch.removeprefix("sm_") + ",code=" + cuda_arch,
        "-Xcompiler=-fPIC",
        "-shared",
        "-I" + str(vendor),
        str(src),
        "-o",
        str(output),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run(cmd, env=env, check=True)
    record = dict(
        format="fissile-build-v1",
        cuda_arch=cuda_arch,
        gpu_kernels_launched=False,
        command=cmd,
        library_sha256=sha256(output),
        compiler=subprocess.check_output(
            [str(compiler), "--version"], text=True, env=env
        ),
        sources={
            p.name: sha256(p)
            for p in [src, vendor / "prompt16.cu", vendor / "qmm_frag.cuh"]
        },
    )
    with output.with_suffix(".build.json").open("x") as f:
        json.dump(record, f, indent=2)
    return record


def build_fp4_gemm(output, cutlass_dir, nvcc=None):
    """CPU compilation of the pinned SM121 CUTLASS NVFP4 GEMM; no GPU kernels.

    ``cutlass_dir`` is an existing CUTLASS source tree containing ``include``
    and ``tools/util/include`` (for example the copy bundled with FlashInfer).
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    cutlass = Path(cutlass_dir).resolve(strict=True)
    includes = [cutlass / "include", cutlass / "tools/util/include"]
    version_header = includes[0] / "cutlass/version.h"
    if not all(p.is_dir() for p in includes) or not version_header.is_file():
        raise ValueError(
            "CUTLASS directory must contain include/ and tools/util/include/"
        )
    version = {
        line.split()[1]: int(line.split()[2])
        for line in version_header.read_text().splitlines()
        if line.startswith("#define CUTLASS_")
        and line.split()[1] in {"CUTLASS_MAJOR", "CUTLASS_MINOR", "CUTLASS_PATCH"}
    }
    compiler = nvcc or shutil.which("nvcc")
    if not compiler:
        raise RuntimeError("CUDA 13 toolchain nvcc is required on PATH")
    src = Path(__file__).parent / "kernels/fp4_gemm.cu"
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(compiler),
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_121a,code=sm_121a",
        "-Xcompiler=-fPIC",
        "-shared",
        *("-I" + str(p) for p in includes),
        str(src),
        "-o",
        str(output),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run(cmd, env=env, check=True)
    record = dict(
        format="fissile-build-v1",
        cuda_arch="sm_121a",
        gpu_kernels_launched=False,
        command=cmd,
        library_sha256=sha256(output),
        compiler=subprocess.check_output(
            [str(compiler), "--version"], text=True, env=env
        ),
        sources={src.name: sha256(src)},
        cutlass_version="{CUTLASS_MAJOR}.{CUTLASS_MINOR}.{CUTLASS_PATCH}".format(
            **version
        ),
        cutlass_version_header_sha256=sha256(version_header),
    )
    with output.with_suffix(".build.json").open("x") as f:
        json.dump(record, f, indent=2)
    return record
