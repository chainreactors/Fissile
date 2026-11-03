"""Check candidate Git files without printing possibly sensitive file contents."""

import hashlib, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    files = sorted(set(raw.decode().split(chr(0))) - {""})
    forbidden_suffixes = {
        ".safetensors",
        ".pt",
        ".pth",
        ".gguf",
        ".so",
        ".o",
        ".a",
        ".whl",
        ".pem",
        ".key",
    }
    patterns = [
        rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        rb"gh[pousr]_[A-Za-z0-9]{30,}",
        rb"github_pat_[A-Za-z0-9_]{30,}",
        rb"AKIA[0-9A-Z]{16}",
        rb"/home/[A-Za-z0-9_.-]+/(?:models|projects)/",
        rb"192\.168\.[0-9]{1,3}\.[0-9]{1,3}",
    ]
    problems = []
    for name in files:
        path = ROOT / name
        if path.suffix in forbidden_suffixes or path.stat().st_size > 2 * 1024 * 1024:
            problems.append(
                {"path": name, "reason": "artifact suffix or oversized file"}
            )
        if path.is_symlink():
            problems.append(
                {"path": name, "reason": "symlink in publication candidate"}
            )
        # The scanner's own regex is not an exposed deployment reference.
        if path.resolve() == Path(__file__).resolve():
            continue
        data = path.read_bytes()
        for index, pattern in enumerate(patterns):
            if re.search(pattern, data):
                problems.append({"path": name, "reason": f"pattern {index}"})
    if problems:
        print(json.dumps({"ok": False, "problems": problems}, indent=2))
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "ok": True,
                "candidate_files": len(files),
                "candidate_bytes": sum((ROOT / f).stat().st_size for f in files),
                "no_model_or_build_artifacts": True,
                "scoped_pattern_scan": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
