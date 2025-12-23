# Long-Horizon Protocol

This repo is designed to validate LLM agents that run for long durations. The goal is not just to complete tasks, but to keep the system healthy and tests passing over time.

## Principles

- Treat `--maximum-coverage` as the health check for long-running agents.
- Run UI checks when UI paths change or after extended runs.
- Avoid hiding regressions behind partial reruns or skipped steps.

## Suggested workflow

1. Start services (Docker or local Reflex + Neo4j).
2. Run a baseline:
   ```bash
   python harness.py --maximum-coverage
   ```
3. Execute the long-horizon agent objective.
4. Repeat `--maximum-coverage` on a schedule or after major changes.
5. If UI interactions are involved, run:
   ```bash
   python harness.py --ui-playwright-check
   ```

## Logging and artifacts

- `MAX_COVERAGE_LOG=1` enables detailed progress logs for long runs.
- The Docker `maxcov` service writes logs to `maxcov_logs/`.
- Coverage artifacts can be collected when `REFLEX_COVERAGE=1` is set.

## Guardrails

- Keep provider keys out of source control.
- Use environment variables for all credentials.
- Prefer reproducible environments (Docker or pinned Python versions).
