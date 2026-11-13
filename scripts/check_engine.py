"""Validate an inference-engine stack for a Fissile backend (docs/engine-upgrades.md).

Run it in an environment with the candidate engine release and its packages
installed; the combination does not have to be validated yet. Arms, each
scoring every input (OUTPUT/ARM/records.jsonl: id, logprobs, elapsed_ms):

  fissile    the backend with --config (always);
  bf16       the plain engine without Fissile (--bf16; SGLang backends);
  perturbed  the backend under SGLang deterministic inference (--perturbed;
             SGLang backends): numerically different, otherwise equivalent.

Checks: installation coverage, repeatability, score_batch bitwise equal to
score (with --activation-scales, SGLang backends) and, against --reference (the
OUTPUT directory of a check of the validated stack on the same inputs):

  bf16_not_worse     BF16 accuracy not significantly lower (McNemar z >= -1.96):
                     the engine itself did not regress;
  fissile_not_worse  Fissile accuracy not significantly lower than the
                     reference's perturbed run (its fissile run if it has none).

A configuration that was evaluated on these inputs with one stack sits on the
favorable side of the near-tie noise, so any numerically different run of it,
including its own perturbed run, scores lower. Comparing with the reference's
perturbed run separates that noise from a real regression. Create a reference
with --bf16 --perturbed.

Writes OUTPUT/report.json: packages, startup and latency per arm, accuracy,
comparisons with the reference and a verdict. Inputs are JSONL rows with id,
input_ids, candidate_token_ids, labels and expected; use the full evaluation
inputs. Inference only.
"""

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_sglang import decode_choice  # noqa: E402  (published decision rule)

RECORDED = ("transformers", "sglang-kernel", "sgl-kernel", "vllm", "sglang")
ARMS = ("fissile", "bf16", "perturbed")


def mcnemar(only_reference, only_candidate):
    n = only_reference + only_candidate
    return (only_candidate - only_reference) / math.sqrt(n) if n else 0.0


def accuracy(records, rows):
    labelled = [r for r in records if "labels" in rows[r["id"]] and "expected" in rows[r["id"]]]
    right = sum(decode_choice(rows[r["id"]]["labels"], r["logprobs"]) == rows[r["id"]]["expected"]
                for r in labelled)
    return dict(labelled=len(labelled), correct=right, accuracy=right / len(labelled) if labelled else None)


def compare_records(records, reference, rows):
    """Candidate records against reference records on the same inputs."""
    ref = {r["id"]: r for r in reference}
    common = [r for r in records if r["id"] in ref]
    if not common:
        raise ValueError("No input in common with the reference")
    diffs = [max(abs(a - b) for a, b in zip(r["logprobs"], ref[r["id"]]["logprobs"], strict=True))
             for r in common]
    out = dict(
        compared=len(common),
        bitwise_equal=sum(d == 0.0 for d in diffs),
        max_logprob_diff=max(diffs),
        median_logprob_diff=statistics.median(diffs),
        mean_ms=statistics.mean(r["elapsed_ms"] for r in common),
        reference_mean_ms=statistics.mean(ref[r["id"]]["elapsed_ms"] for r in common),
    )
    labelled = [r for r in common if "labels" in rows[r["id"]] and "expected" in rows[r["id"]]]
    if labelled:
        def decide(logprobs, row):
            return decode_choice(row["labels"], logprobs)

        new = {r["id"]: decide(r["logprobs"], rows[r["id"]]) for r in labelled}
        old = {r["id"]: decide(ref[r["id"]]["logprobs"], rows[r["id"]]) for r in labelled}
        right_new = {i for i in new if new[i] == rows[i]["expected"]}
        right_old = {i for i in old if old[i] == rows[i]["expected"]}
        out.update(
            labelled=len(labelled),
            decisions_changed=sum(new[i] != old[i] for i in new),
            accuracy=len(right_new) / len(labelled),
            reference_accuracy=len(right_old) / len(labelled),
            only_reference_correct=len(right_old - right_new),
            only_candidate_correct=len(right_new - right_old),
            mcnemar_z=mcnemar(len(right_old - right_new), len(right_new - right_old)),
        )
    return out


def compare_with_reference(arms, reference_arms, rows):
    """Comparisons used by the verdict, plus informational ones.

    ``arms``/``reference_arms`` map arm name to records. Returns a dict of
    comparisons keyed "candidate_arm_vs_reference_arm" and the gate baseline."""
    out = {}
    for arm, ref_arm in (("bf16", "bf16"), ("fissile", "perturbed"), ("fissile", "fissile"),
                         ("perturbed", "perturbed")):
        if arm in arms and ref_arm in reference_arms:
            out[f"{arm}_vs_{ref_arm}"] = compare_records(arms[arm], reference_arms[ref_arm], rows)
    gate = "fissile_vs_perturbed" if "fissile_vs_perturbed" in out else (
        "fissile_vs_fissile" if "fissile_vs_fissile" in out else None)
    return out, gate


def verdict(report, *, z_limit=-1.96):
    """Checks that must pass for the stack to be added to engines.VALIDATED."""
    checks = dict(installed=report["install"]["complete"],
                  repeatable=report["repeat"]["equal"] == report["repeat"]["scored"])
    if "batch" in report:
        checks["batch_exact"] = report["batch"]["equal"] == report["batch"]["scored"]
    reference = report.get("reference")
    if reference:
        comparisons = reference["comparisons"]
        if "accuracy" in comparisons.get("bf16_vs_bf16", {}):
            checks["bf16_not_worse"] = comparisons["bf16_vs_bf16"]["mcnemar_z"] >= z_limit
        gate = reference.get("gate")
        if gate and "accuracy" in comparisons[gate]:
            checks["fissile_not_worse"] = comparisons[gate]["mcnemar_z"] >= z_limit
    return dict(checks=checks, passed=all(checks.values()))


def phase_config(config, output, phase):
    """Copy of the configuration with this phase's report directory. Relative
    paths in a configuration resolve against its directory, so every path is
    made absolute."""
    from fissile.config import Config

    data = Config.load(config, verify_files=False).as_dict()
    data["report_dir"] = str(output.resolve() / f"reports-{phase}")
    path = output / f"fissile-{phase}.json"
    path.write_text(json.dumps(data, indent=2))
    return path, Path(data["report_dir"])


def install_report(report_dir):
    install = json.loads((report_dir / "quant-install.json").read_text())
    coverage = report_dir / "first-forward-coverage.json"
    complete = bool(install.get("complete", True))
    if coverage.exists():
        complete = complete and json.loads(coverage.read_text()).get("complete") is True
    keys = ("engine", "engine_version", "engine_validated", "kernel_provider", "readout")
    return dict(complete=complete, **{k: install[k] for k in keys if k in install})


@contextmanager
def deterministic_sglang():
    """SGLang engines created inside run with deterministic inference."""
    import sglang

    engine = sglang.Engine
    sglang.Engine = lambda *a, **kw: engine(*a, **{**kw, "enable_deterministic_inference": True})
    try:
        yield
    finally:
        sglang.Engine = engine


def score_arm(session, rows, directory, *, warmup, repeat):
    directory.mkdir()
    for row in rows[:warmup]:
        session.score(row["input_ids"], row["candidate_token_ids"])
    records = []
    with (directory / "records.jsonl").open("x") as sink:
        for i, row in enumerate(rows):
            t = time.perf_counter()
            result = session.score(row["input_ids"], row["candidate_token_ids"])
            record = dict(id=row["id"], logprobs=list(result.token_logprobs),
                          elapsed_ms=(time.perf_counter() - t) * 1000)
            records.append(record)
            sink.write(json.dumps(record) + "\n")
            if (i + 1) % 1000 == 0:
                sink.flush()
                (directory / "progress.json").write_text(json.dumps(dict(done=i + 1, total=len(rows))))
    again = [list(session.score(r["input_ids"], r["candidate_token_ids"]).token_logprobs) for r in rows[:repeat]]
    repeated = dict(scored=len(again), equal=sum(a == r["logprobs"] for a, r in zip(again, records)))
    return records, repeated


def load_reference(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    arms = {}
    for arm in ARMS:
        path = directory / arm / "records.jsonl"
        if path.exists():
            arms[arm] = [json.loads(line) for line in path.open()]
    if "fissile" not in arms:
        raise SystemExit(f"Reference {directory} has no fissile/records.jsonl")
    return report, arms


def run(args):
    import importlib.metadata
    from fissile.backends.engines import engine_status
    from fissile.backends.registry import get_backend, open_backend

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    spec = get_backend(args.backend)
    if spec.engine is None:
        raise SystemExit(f"{args.backend} runs no external engine; nothing to validate")
    if (args.bf16 or args.perturbed or args.activation_scales) and spec.engine != "sglang":
        raise SystemExit("--bf16, --perturbed and --activation-scales need an SGLang backend")
    if not 2 <= args.batch_size <= 16:
        raise SystemExit("--batch-size must be 2-16")
    rows = [json.loads(line) for line in open(args.inputs)]
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise SystemExit("Input ids must be unique")
    reference = load_reference(args.reference) if args.reference else None

    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    report = dict(backend=args.backend, inputs=len(rows), **engine_status(spec.engine),
                  recorded={p: version(p) for p in RECORDED if version(p)})
    options = dict(allow_unvalidated_engine=True)
    arms, timing = {}, {}

    def fissile_arm(name, context):
        config, report_dir = phase_config(args.config, output, name)
        started = time.perf_counter()
        with context(), open_backend(args.backend, config, args.model, **options) as session:
            timing[name] = dict(startup_seconds=round(time.perf_counter() - started, 1))
            arms[name], repeated = score_arm(session, rows, output / name, warmup=args.warmup, repeat=args.repeat)
        return install_report(report_dir), repeated

    report["install"], report["repeat"] = fissile_arm("fissile", nullcontext)
    if args.perturbed:
        install, repeated = fissile_arm("perturbed", deterministic_sglang)
        report["perturbed"] = dict(install=install, repeat=repeated)
    if args.bf16:
        from benchmark_sglang import baseline_session

        started = time.perf_counter()
        with baseline_session(args.model) as session:
            timing["bf16"] = dict(startup_seconds=round(time.perf_counter() - started, 1))
            arms["bf16"], repeated = score_arm(session, rows, output / "bf16", warmup=args.warmup, repeat=args.repeat)
        report["bf16"] = dict(repeat=repeated)
    for name, records in arms.items():
        timing[name]["mean_ms"] = statistics.mean(r["elapsed_ms"] for r in records)
        timing[name].update(accuracy(records, by_id))
    report["arms"] = timing
    if args.activation_scales:
        config, report_dir = phase_config(args.config, output, "batch")
        subset = rows[: args.batch_inputs]
        with open_backend(args.backend, config, args.model, batch=True,
                          activation_scales=args.activation_scales, **options) as session:
            single = [list(session.score(r["input_ids"], r["candidate_token_ids"]).token_logprobs)
                      for r in subset]
            batched = []
            for i in range(0, len(subset), args.batch_size):
                group = subset[i : i + args.batch_size]
                batched += [list(s.token_logprobs) for s in session.score_batch(
                    [(r["input_ids"], r["candidate_token_ids"]) for r in group])]
        report["batch"] = dict(scored=len(subset), equal=sum(a == b for a, b in zip(single, batched)),
                               batch_size=args.batch_size, install=install_report(report_dir))
    if reference:
        ref_report, ref_arms = reference
        comparisons, gate = compare_with_reference(arms, ref_arms, by_id)
        report["reference"] = dict(
            directory=str(Path(args.reference).resolve()),
            engine_version=ref_report.get("engine_version"), stack=ref_report.get("stack"),
            arms=ref_report.get("arms"), comparisons=comparisons, gate=gate)
    report["verdict"] = verdict(report)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"]))
    return 0 if report["verdict"]["passed"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", required=True, help="sglang, sglang-candidates or vllm")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--inputs", required=True, help="Labelled inputs; the full evaluation set for accuracy checks")
    p.add_argument("--output", required=True, help="New directory")
    p.add_argument("--reference", help="OUTPUT directory of a check of the validated stack on the same inputs")
    p.add_argument("--bf16", action="store_true", help="Also score the plain engine (SGLang backends)")
    p.add_argument("--perturbed", action="store_true",
                   help="Also score under deterministic inference (SGLang backends); needed in references")
    p.add_argument("--activation-scales", help="Static scale file: enables the batch check (SGLang backends)")
    p.add_argument("--batch-size", type=int, default=4, help="Inputs per score_batch call (2-16)")
    p.add_argument("--batch-inputs", type=int, default=256, help="Inputs scored in the batch check")
    p.add_argument("--warmup", type=int, default=48)
    p.add_argument("--repeat", type=int, default=64)
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
