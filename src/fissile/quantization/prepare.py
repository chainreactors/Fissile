"""Offline preparation of one projection using a full input second moment.

Caller supplies effective merged BF16 weights and representative unlabeled moments.
No training, dataset download, labels, or implicit model evaluation.
"""

from pathlib import Path
from fissile.config import sha256, require_gpu


def prepare_projection(weight, hessian, output):
    import torch
    from safetensors.torch import save_file
    from fissile.kernels.packs import weight_hash
    from .mse import quantize_weight as grid
    from .gptq_stable import prepare_inverse, quantize_weight

    require_gpu()
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Disable TF32 before preparation for IEEE FP32 updates")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with torch.inference_mode():
        inverse = prepare_inverse(hessian, damping=0.01, act_order=False)
        q, s, g = quantize_weight(weight, inverse, grid(weight))
        output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "q": q.view(torch.uint8).cpu(),
                "s": s.view(torch.uint8).cpu(),
                "g": g.cpu(),
            },
            str(output),
        )
    return {
        "path": output.name,
        "shape": list(weight.shape),
        "sha256": sha256(output),
        "effective_weight_sha256": weight_hash(weight),
    }
