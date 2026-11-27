# Contributing

## Setup and tests

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The core package and its tests run on CPU without Torch. GPU work needs an
engine environment ([setup](docs/artifacts.md#sglang-environment)).

## Changes

- Keep numerical changes separate from packaging or engine integration changes.
- State the model, hardware and workload a change supports. Raise an explicit
  error for an unsupported shape or backend instead of silently falling back.
- For numerical or performance changes, report raw-score equality or accuracy,
  projection coverage and latency, measured on an idle GPU with the current
  code.
- Register only the `(profile, recipe, provider)` combinations a backend has
  been verified with; see [Extending JevInfer](docs/extensions.md).

## Architecture boundaries

Core and model metadata use only the standard library. Kernels do not import
engine or model code. Providers own numerical implementations; the runtime owns
weight identity, binding and execution coverage; engine adapters own module
layout and engine lifecycle. Offline quantization does not depend on a running
engine. Model profiles and precision recipes stay separate.

## Before publishing

Run `python scripts/check_release.py` and build a wheel. Do not commit model
weights, fine-tuning adapters, prompts, local configurations or compiled
libraries.

## Documentation

README covers usage and measured performance; `docs/` holds configuration and
development guides; `bench/` holds benchmark results with their method and data.
