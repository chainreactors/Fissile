"""Calibrate static NVFP4 activation scales for the gb10-mixed-v2-static provider.

Scores every N-th row of a tokenized JSONL file (fields ``input_ids`` and
``candidate_token_ids``) with a config selecting ``gb10-mixed-v2-calibration``,
which keeps the gb10-mixed-v2 arithmetic and records each activation site's largest
per-forward global scale. Writes ``margin`` x that maximum per site as a
``fissile-activation-scales-v1`` file. No training occurs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Fissile config, kernel_provider gb10-mixed-v2-calibration")
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--every", type=int, default=29, help="Score rows 0, N, 2N, ...")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--backend", default="sglang-candidates")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.every < 1 or not args.margin >= 1.0:
        raise ValueError("--every must be >= 1 and --margin >= 1")

    from fissile.backends.registry import open_backend
    from fissile.config import Config
    from fissile.providers.static_scales import (
        CALIBRATION_FORMAT,
        CALIBRATION_OUTPUT,
        SCALES_FORMAT,
    )

    config = Config.load(args.config)
    if config.kernel_provider != "gb10-mixed-v2-calibration":
        raise ValueError("The config must select gb10-mixed-v2-calibration")
    with open(args.inputs) as f:
        rows = [json.loads(line) for i, line in enumerate(f) if i % args.every == 0]
    with tempfile.TemporaryDirectory() as scratch:
        record_path = Path(scratch) / "calibration.json"
        os.environ[CALIBRATION_OUTPUT] = str(record_path)
        with open_backend(args.backend, args.config, args.model) as session:
            for row in rows:
                session.score(row["input_ids"], row["candidate_token_ids"])
        record = json.loads(record_path.read_text())
    if record.get("format") != CALIBRATION_FORMAT or record["forwards"] < len(rows):
        raise RuntimeError("Calibration record is incomplete")
    with open(args.inputs, "rb") as f:
        inputs_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    scales = {
        "format": SCALES_FORMAT,
        "profile": config.profile,
        "recipe": config.recipe,
        "margin": args.margin,
        "calibration": {
            "inputs_sha256": inputs_sha256,
            "every": args.every,
            "rows": len(rows),
            "tokens": sum(len(r["input_ids"]) for r in rows),
            "forwards": record["forwards"],
        },
        "scales": {site: args.margin * g for site, g in sorted(record["max_global_scale"].items())},
    }
    with output.open("x") as f:
        json.dump(scales, f, indent=1)
        f.write("\n")
    print(json.dumps({k: v for k, v in scales.items() if k != "scales"} | {"sites": len(scales["scales"])}))


if __name__ == "__main__":
    main()
