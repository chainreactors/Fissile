"""Input second moments of projection groups from native BF16 forwards.

Each forward contributes at most ``max_rows`` evenly spaced input rows per
group as an FP32 Gram product, accumulated in FP64.
"""

from functools import partial


def collect_moments(modules, groups, run, *, max_rows=256):
    """Mean FP64 input second moment of each group, on the modules' device.

    ``modules`` maps canonical keys to projection modules; each group in
    ``groups`` is a tuple of keys consuming one input. ``run()`` executes every
    calibration forward. Returns ``(moments, rows_per_group)`` keyed by group.
    """
    import torch

    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Disable TF32 for IEEE FP32 Gram products")
    if not 1 <= max_rows <= 256:
        raise ValueError("FP32 Gram products accumulate at most 256 rows")
    sums, counts = {}, {}

    def accumulate(module, inputs, *, group):
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
        if not x.is_cuda or x.dtype != torch.bfloat16:
            raise ValueError("Expected CUDA BF16 projection inputs: " + group[0])
        if len(x) > max_rows:
            positions = torch.linspace(0, len(x) - 1, max_rows, device=x.device, dtype=torch.long)
            x = x.index_select(0, positions)
        x = x.float()
        moment = x.t() @ x
        if not bool(torch.isfinite(moment).all()):
            raise ValueError("Nonfinite input moment: " + group[0])
        if group not in sums:
            exact = x.double().t() @ x.double()
            error = float((moment.double() - exact).norm() / exact.norm().clamp_min(1.0e-30))
            if error > 1.0e-4:
                raise ValueError("FP32 Gram product is insufficiently accurate: " + group[0])
            sums[group] = moment.double()
            counts[group] = 0
        else:
            sums[group].add_(moment)
        counts[group] += len(x)

    handles = [modules[g[0]].register_forward_pre_hook(partial(accumulate, group=g)) for g in groups]
    try:
        run()
    finally:
        for handle in handles:
            handle.remove()
    missing = [g[0] for g in groups if g not in sums]
    if missing:
        raise ValueError("Calibration forwards did not reach: " + ", ".join(missing))
    with torch.inference_mode():
        for group in groups:
            sums[group].div_(counts[group])
    return sums, counts
