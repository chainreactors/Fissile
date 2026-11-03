"""Padded-row masking for SGLang prefill CUDA graphs.

Graph replay pads the token count to a captured bucket and leaves stale rows from
earlier requests in the padding. NVFP4 activation quantization uses a tensor-wide
amax, so during capture every projection input keeps only rows below the live
token count, which the eager outer forward publishes before each replay. Outside
capture the mask is the identity, so eager execution is unchanged.
"""

import torch


class ValidRows:
    def __init__(self):
        self.count = None

    def allocate(self, device):
        if self.count is None:
            self.count = torch.zeros((), dtype=torch.int64, device=device)

    def publish(self, n):
        if self.count is None:
            raise RuntimeError("Valid-row counter was not allocated at weight load")
        self.count.fill_(int(n))

    def mask(self, x):
        if not torch.cuda.is_current_stream_capturing():
            return x
        if self.count is None or self.count.device != x.device:
            raise RuntimeError("Valid-row counter unavailable during graph capture")
        rows = torch.arange(x.shape[0], device=x.device)
        return torch.where((rows < self.count)[:, None], x, 0.0)


VALID_ROWS = ValidRows()
