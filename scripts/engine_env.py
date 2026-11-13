"""Create a Python environment with an inference-engine release for validation.

usage: python scripts/engine_env.py --engine sglang==0.5.23 --output DIR [--torch-index URL]

Creates a virtual environment with the engine release (which pins torch, Triton
and its kernel packages), makes sure torch is the CUDA build from --torch-index,
installs Fissile from this repository without dependencies and writes
DIR/stack.json with the installed engine and STACK versions and whether they
form a validated stack. Package indexes, mirrors and proxies come from the
caller's pip configuration (PIP_INDEX_URL, pip.conf, HTTPS_PROXY); the only
index added is the PyTorch CUDA wheel index. Then validate the environment with
scripts/check_engine.py (docs/engine-upgrades.md).
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
ENGINES = ("sglang", "vllm")


def plan(engine, output, *, torch_index=TORCH_INDEX, fissile=ROOT, python=sys.executable):
    """Commands that build the environment (lists of arguments)."""
    name = engine.split("==", 1)[0]
    if name not in ENGINES or "==" not in engine:
        raise ValueError(f"--engine must be one of {', '.join(ENGINES)} with an exact release, e.g. sglang==0.5.23")
    pip = [str(Path(output) / "bin/python"), "-m", "pip"]
    return [
        [python, "-m", "venv", str(output)],
        pip + ["install", "--upgrade", "pip"],
        pip + ["install", "--extra-index-url", torch_index, engine],
        pip + ["install", "--no-deps", str(fissile)],
    ]


STATUS = """
import json, sys
from fissile.backends.engines import engine_status
print(json.dumps(engine_status(sys.argv[1])))
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True, help="Exact release, e.g. sglang==0.5.23 or vllm==0.33.0")
    p.add_argument("--output", required=True, help="New directory for the virtual environment")
    p.add_argument("--torch-index", default=TORCH_INDEX, help="PyTorch wheel index of the CUDA build")
    a = p.parse_args()
    output = Path(a.output).resolve()
    if output.exists():
        raise SystemExit(f"{output} exists")
    for command in plan(a.engine, output, torch_index=a.torch_index):
        subprocess.run(command, check=True)
    python = str(output / "bin/python")
    torch = subprocess.run([python, "-c", "import torch; print(torch.__version__, torch.version.cuda)"],
                           check=True, capture_output=True, text=True).stdout.split()
    if len(torch) < 2 or torch[1] == "None":
        # The engine resolved a CPU build: reinstall the same torch release from the CUDA index.
        subprocess.run([python, "-m", "pip", "install", "--force-reinstall", "--no-deps",
                        "--index-url", a.torch_index, f"torch=={torch[0].split('+')[0]}"], check=True)
    status = json.loads(subprocess.run([python, "-c", STATUS, a.engine.split("==")[0]],
                                       check=True, capture_output=True, text=True).stdout)
    (output / "stack.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
