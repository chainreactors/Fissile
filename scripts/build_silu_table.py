"""Build a mathematical BF16 SiLU LUT; needs an idle admitted GB10 device."""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    from fissile.config import require_gpu, sha256

    require_gpu()
    from fissile.kernels.paired_glu import _build_silu_table
    from safetensors.torch import save_file
    import torch

    with torch.inference_mode():
        _, table = _build_silu_table()
        output.parent.mkdir(parents=True, exist_ok=True)
        save_file({"bits": table.view(torch.int16).cpu()}, str(output))
    print(sha256(output))


if __name__ == "__main__":
    main()
