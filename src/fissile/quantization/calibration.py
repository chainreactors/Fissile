"""Calibration inputs: one tokenized, unlabeled input per JSONL line."""

import json
from pathlib import Path


def read_calibration(path, profile):
    """Token-ID lists of every non-empty line; other fields are ignored."""
    rows = []
    with Path(path).open() as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            ids = row.get("input_ids") if isinstance(row, dict) else None
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= profile.max_context
                or any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < profile.vocab_size for t in ids)
            ):
                raise ValueError(
                    f"Calibration line {number}: expected input_ids with 1 to "
                    f"{profile.max_context} token IDs below {profile.vocab_size}"
                )
            rows.append(ids)
    if not rows:
        raise ValueError("Calibration file has no inputs")
    return rows
