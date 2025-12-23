# long-horizon-llm-harness

This repository is a long-horizon LLM engineering harness. It is intentionally
named like a resume builder but the real mission is to facilitate the design,
training, and testing of LLM agents that must operate reliably for hours, days,
weeks, or longer without human intervention. The harness emphasizes strict
verification, high test coverage, and structured process control so that agents
stay on track and do not regress.

## Why this exists

Long-running autonomous coding agents often fail due to requirement drift,
missed specs, or weak self-verification. This repo provides a disciplined
workflow, strong automation, and coverage-driven checks to push beyond those
limits. It is fit-for-service because the test suite is extensive and the
maximum-coverage mode targets >90% code coverage. UI coverage is being expanded
in `scripts/`, which will further strengthen long-run reliability.

## Key capabilities

- **Long-horizon harness protocol**: `AGENTS.md` defines a finite-state
  workflow with strict gates, requirement tracking, and persistent memory.
- **Coverage-driven simulation**: `--maximum-coverage` exercises UI state,
  DB paths, PDF generation, and failure handling.
- **End-to-end orchestration**: `--run-all-tests` runs coverage, verifies a
  clean Reflex startup, and executes Playwright UI checks.
- **Typst-based PDF generation**: highly controlled layout pipeline for
  deterministic resume output.
- **Extensible automation**: scripts in `scripts/` add UI and long-run checks.

## Project structure

- `AGENTS.md`: Long-horizon protocol and gates for autonomous agents.
- `harness.py`: CLI entrypoint for tests, coverage, and PDF generation.
- `lib.typ`: Typst resume template and layout logic.
- `docs/`: Protocol state, requirements, plans, and logs.
- `scripts/`: UI and max-coverage automation (actively evolving).

## Quick start

```bash
# Run maximum-coverage simulation
python harness.py --maximum-coverage

# Run the full test gate
python harness.py --run-all-tests

# Compile a PDF for the current resume data
python harness.py --compile-pdf /tmp/preview.pdf
```

## Long-horizon guidance

This repo is designed for long-run autonomous agent work. If you are using an
agent (Codex, Gemini CLI, Cursor, etc.), read and follow `AGENTS.md`. It defines
the required lifecycle (Intake -> Requirements -> Design -> Implementation ->
Verification -> Stabilization -> Maintenance), and mandates that
`python harness.py --run-all-tests` passes before completion.

## Test and coverage notes

See `docs/TESTING.md` for detailed paths and environment variables. Highlights:

- `--maximum-coverage` is the primary simulation path.
- `--run-all-tests` is the strict gate for completion.
- UI and PDF checks are handled via Playwright and Typst.

## Docker and CI safety

`scripts/run_maxcov_e2e.sh --force` performs a full Docker reset (containers,
images, volumes, networks). This is safe only on dedicated runners or isolated
dev machines. Do not run it in a shared CI job with service containers unless
you want them removed.

## Status

UI automation in `scripts/` is actively evolving. The long-horizon harness is
stable and focused on maximum coverage, reproducibility, and strict gatekeeping.

## License

See `LICENSE`.
