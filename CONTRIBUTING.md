# Contributing

Thanks for contributing to the Long-Horizon LLM Harness.

## Ground rules

- Keep long-horizon reliability as the top priority.
- Do not commit secrets or keys.
- Avoid touching `scripts/` while UI tests are being updated by other processes.

## Development workflow

1. Create a feature branch.
2. Make focused changes and add docs for new behavior.
3. Run the relevant tests:
   - `python harness.py --maximum-coverage`
   - `python harness.py --ui-playwright-check` (if UI changes)
4. Open a PR with a clear summary and test evidence.

## Code style

- Keep changes minimal and explicit.
- Prefer small, composable functions.
- Add comments only when logic is non-obvious.

## Reporting issues

Use the issue templates in `.github/ISSUE_TEMPLATE` or start a Discussion for open-ended questions.
