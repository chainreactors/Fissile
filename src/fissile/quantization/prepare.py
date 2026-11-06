"""Offline ``full-h16-order`` preparation of projection packs.

Block scales are chosen against each block's 16x16 input moment, then GPTQ
compensates in descending input-moment order with 1% damping. Callers supply
effective BF16 weights and unlabeled input moments; no labels, training or
model evaluation.
"""

from pathlib import Path
from fissile.config import sha256, require_gpu

METHOD = "full-h16-order"


def prepare_moment(hessian):
    """GPTQ inverse and 16x16 block moments of one full input moment.

    Shared by every projection that consumes the same input.
    """
    import torch
    from .gptq_stable import prepare_inverse
    from .hessian import local_moments

    with torch.inference_mode():
        return prepare_inverse(hessian, damping=0.01, act_order=True), local_moments(hessian)


def prepare_projection(weight, hessian, output, *, prepared=None):
    """Write the pack of one projection and return its manifest entry.

    ``hessian`` is the full CUDA input second moment of ``weight``;
    ``prepared`` may pass a precomputed ``prepare_moment(hessian)``.
    """
    import torch
    from safetensors.torch import load_file, save_file
    from fissile.kernels.packs import weight_hash
    from .gptq_stable import quantize_weight
    from .hessian import quantize_weight as scale_grid

    require_gpu()
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Disable TF32 before preparation for IEEE FP32 updates")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with torch.inference_mode():
        inverse, local = prepared if prepared is not None else prepare_moment(hessian)
        packed = quantize_weight(weight, inverse, scale_grid(weight, local))
        tensors = {
            "q": packed[0].view(torch.uint8).cpu(),
            "s": packed[1].view(torch.uint8).cpu(),
            "g": packed[2].cpu(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(output))
        restored = load_file(str(output))
        if set(restored) != set(tensors) or any(
            not torch.equal(restored[k].reshape(-1).view(torch.uint8), tensors[k].reshape(-1).view(torch.uint8))
            for k in tensors
        ):
            raise ValueError("Serialized pack does not reproduce the prepared tensors")
    return {
        "path": output.name,
        "shape": list(weight.shape),
        "sha256": sha256(output),
        "effective_weight_sha256": weight_hash(weight),
    }
