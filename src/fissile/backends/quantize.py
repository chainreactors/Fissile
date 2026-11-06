"""``fissile quantize``: calibrate a Transformers BF16 model and prepare its packs.

Module names and input groups come from the profile's engine layout; the
numerical work is in ``fissile.quantization``.
"""

import os
from pathlib import Path

from fissile.config import sha256
from fissile.models import DEFAULT_PROFILE
from fissile.recipes import DEFAULT_RECIPE


def load_model(model_dir, profile=DEFAULT_PROFILE):
    """Load a local BF16 model with the Transformers class of the profile's layout."""
    import torch
    import transformers
    from .layouts import get_layout

    cls = getattr(transformers, get_layout(profile).hf_class)
    model = cls.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
        local_files_only=True,
        trust_remote_code=False,
    )
    return model.eval().requires_grad_(False)


def release_page_cache(model_dir):
    """Drop clean page cache of the checkpoint files after loading. On unified
    memory devices (GB10) CUDA counts only free memory as available."""
    if not hasattr(os, "posix_fadvise"):
        return
    for path in Path(model_dir).rglob("*.safetensors"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def quantize_model(model, rows, packs_dir, manifest, *, profile=DEFAULT_PROFILE, recipe=DEFAULT_RECIPE,
                   max_rows=256, calibration_sha256=None, progress=None):
    """Collect input moments of a loaded, frozen BF16 model over ``rows``
    (token-ID lists), then prepare every pack and write the manifest."""
    import torch
    from fissile.quantization.moments import collect_moments
    from fissile.quantization.packs import check_outputs, prepare_packs
    from .layouts import input_groups
    from .torch import modules_for

    selected, _ = check_outputs(profile, recipe, packs_dir, manifest)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Requires model.eval().requires_grad_(False)")
    shapes = selected.shapes()
    modules = modules_for(model, selected)
    for key, module in modules.items():
        w = module.weight
        if getattr(module, "bias", None) is not None or w.dtype != torch.bfloat16 or list(w.shape) != shapes[key]:
            raise ValueError("Expected a bias-free BF16 projection of the profile shape: " + key)
    groups = input_groups(selected)
    device = modules[selected.head].weight.device

    def run():
        # Final-token scoring: the output head sees the last position only.
        with torch.inference_mode():
            for index, ids in enumerate(rows):
                model(input_ids=torch.tensor([ids], device=device), use_cache=False, logits_to_keep=1)
                if progress:
                    progress("calibrate", index + 1, len(rows))

    moments, counts = collect_moments(modules, groups, run, max_rows=max_rows)
    calibration = dict(sha256=calibration_sha256, inputs=len(rows), max_rows_per_forward=max_rows,
                       rows_per_group={g[0]: counts[g] for g in groups})
    weights = {key: module.weight.detach() for key, module in modules.items()}
    return prepare_packs(weights, moments, groups, packs_dir, manifest, profile=selected.name, recipe=recipe,
                         calibration=calibration, progress=progress)


def quantize(model_dir, calibration, packs_dir, manifest, *, profile=DEFAULT_PROFILE, recipe=DEFAULT_RECIPE,
             max_rows=256, progress=None):
    """Validate inputs, load the model, release its checkpoint page cache, then
    ``quantize_model``. TF32 is disabled for IEEE FP32 products."""
    import torch
    from fissile.quantization.calibration import read_calibration
    from fissile.quantization.packs import check_outputs

    selected, _ = check_outputs(profile, recipe, packs_dir, manifest)
    rows = read_calibration(calibration, selected)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(model_dir, selected.name)
    release_page_cache(model_dir)
    return quantize_model(model, rows, packs_dir, manifest, profile=selected.name, recipe=recipe,
                          max_rows=max_rows, calibration_sha256=sha256(calibration), progress=progress)
