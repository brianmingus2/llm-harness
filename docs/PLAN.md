# PLAN

## Milestones
- [x] M1 (REQ-001): Update static analysis tooling to add codespell, pip-audit, safety, and pyupgrade with safe parsing and non-destructive pyupgrade checks.
  - Exit criteria: tools appear in the static report; pyupgrade runs on temp copies only.
- [x] M2 (REQ-002, REQ-003, REQ-005): Remove static-analysis timeouts for pip-audit/pytype and fail on missing tools.
  - Exit criteria: pip-audit and pytype run without subprocess timeout; missing tool status=fail and run-all-tests exits non-zero if any fail.
- [x] M3 (REQ-006): Wire diagram generation into the full test flow and ensure dependencies are installed.
  - Exit criteria: diagrams render into diagrams/ during run-all-tests and any failures are reported.
- [x] M4 (REQ-007, REQ-008): Update Docker e2e runner to execute --run-all-tests with all dependencies installed.
  - Exit criteria: scripts/run_maxcov_e2e.sh runs run-all-tests and Dockerfile installs required tools.
- [x] M5 (REQ-004): Run python harness.py --run-all-tests; record results and update state files.
  - Exit criteria: run-all-tests passes; docs/TEST_LOG.md and docs/AGENT_STATE.md updated.
- [x] M6 (REQ-009, REQ-010): Replace brian_mingus_resume.json with michael_scott_resume.json and seed full resume data (ASCII-only) across profile, experience, education, founder roles, skills, and custom sections.
  - Exit criteria: default asset references point to michael_scott_resume.json; new JSON validated with required fields and no non-ASCII content.
- [x] M7 (REQ-011, REQ-012): Add safe import guardrails and ensure dev/test flows use stub or ephemeral DB; document CLI import workflow.
  - Exit criteria: importing assets refuses to overwrite existing resume without explicit override; docs updated with import usage and test isolation notes.
- [x] M8 (REQ-013): Run python harness.py --run-all-tests and record results for the updated resume assets.
  - Exit criteria: run-all-tests passes; docs/TEST_LOG.md, docs/AGENT_STATE.md, and docs/FEATURES.json updated with evidence.
- [x] M9 (REQ-014): Compile the Michael Scott PDF with a stubbed DB and review the rendered output via Vision.
  - Exit criteria: PDF compiled with MAX_COVERAGE_STUB_DB=1, rendered pages reviewed, and evidence recorded in docs/TEST_LOG.md.
- [x] M10 (REQ-015): Adjust Typst header layout to keep long names on a single line.
  - Exit criteria: long names render without wrapping; run-all-tests passes; evidence logged in docs/TEST_LOG.md.

## Dependencies
- Python tooling installed for all static analyzers in harness.py.
- System packages for PDF inspection (pdftotext, mutool) and graphviz.
- Network access for vulnerability scanners (pip-audit, safety).
- Docker image must include run-all-tests dependencies.
- Target Python version for pyupgrade: 3.12 (from runtime).
- New resume asset JSON must match the expected import schema.

## Risks
- Vulnerability scanners may require API keys or network connectivity.
- pyupgrade may modify files if not isolated.
- run-all-tests is long; failures must be fixed before completion.
- Diagram generation tools can fail if system dependencies are missing.
- Expanded resume content may affect PDF length; ensure UI/PDF tests still pass.
- Import guardrails must not block intended CLI usage for new users with empty DBs.

## Rollback
- Revert harness.py, requirements.txt, Dockerfile, and scripts/run_maxcov_e2e.sh if run-all-tests fails after changes.
