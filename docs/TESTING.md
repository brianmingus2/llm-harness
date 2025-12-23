# Testing

This repo is designed to validate long-horizon behavior. Testing is built around coverage-driven simulations and UI automation.

## Core paths

```bash
# End-to-end maximum-coverage simulation (skips LLM calls by default)
python harness.py --maximum-coverage

# Run the Reflex coverage server and drive it via Playwright
python harness.py --maximum-coverage-reflex

# UI traversal check (requires a running Reflex app)
python harness.py --ui-playwright-check
```

## Docker-based runs

```bash
# Runs the max coverage mode in a container
./scripts/run_maxcov.sh
```

Note: `scripts/` is actively evolving for UI coverage and long-run checks. Avoid changing those files while tests are running.

## Coverage notes

- `--maximum-coverage` exercises UI state transitions, DB paths, and PDF branches.
- `REFLEX_COVERAGE=1` enables per-worker coverage tracking.
- `MAX_COVERAGE_LOG=1` emits detailed progress logs (useful for long runs).
- `--run-all-tests` runs the UI flow with `MAX_COVERAGE_STUB_DB=1` to avoid touching user data.

## Failure path simulation

The maximum-coverage mode supports failure simulation for DB, LLM, and PDF paths. See the CLI help in `harness.py` for toggles such as:

- `--maximum-coverage-failures`
- `MAX_COVERAGE_FORCE_DB_ERROR_ON_LOAD=1`
- `MAX_COVERAGE_SKIP_PDF=1`

## UI automation

The Playwright checks validate the UI and PDF embed surface. Use these environment variables as needed:

- `PLAYWRIGHT_URL`
- `REFLEX_URL`
- `REFLEX_APP_URL`

## Suggested long-run cadence

For multi-hour or multi-day runs:

1. Run a clean `--maximum-coverage` baseline.
2. Execute the agent objective.
3. Repeat `--maximum-coverage` every N hours or after major changes.
4. Run UI checks for regression coverage when the UI surface changes.
