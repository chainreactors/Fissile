# Contributing

Keep numerical changes separate from packaging or engine integration changes.
State the model, hardware and workload a change supports. Add explicit checks
instead of silently selecting a fallback for an unsupported shape or backend.

Dependency boundaries are tested: core and model metadata use the standard
library; kernels must not import engine/model code; runtime owns reusable
execution; adapters own engine layout and lifecycle; offline quantization does
not depend on an active engine. Keep GPU work behind explicit calls.

Run CPU tests with `python -m unittest discover -s tests -v` and build a wheel.
Run `python scripts/check_release.py` before publishing. Do not commit model
weights, adapters, prompts, local configs or compiled kernels.

Keep README focused on usage and measured performance; put stable configuration
and development guides in `docs/`, and sanitized benchmark measurements with
their method in `bench/`. Machine-specific deployment manuals, run logs and
one-off extraction or refactoring reports belong outside this repository.

GPU checks require an idle device and the local operator's scheduling policy.
No contributor workflow should kill another job or dispatch remotely by default.
Record numerical equality, layer/projection coverage, and accuracy alongside
performance. Historical results are not qualification for a changed package.

New engines implement the checked scoring contract and register exact capability
combinations; see `docs/extensions.md`. Do not broaden qualification merely by
adding a registry name. Keep model metadata and precision recipes separate.
Providers own numerical implementations; runtime owns identity and coverage.
