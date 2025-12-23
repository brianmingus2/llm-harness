# AGENT_STATE

## Status
- Phase: Stabilization
- Owner: Codex
- Timestamp: 2025-12-23T12:21:30Z

## Baseline Assumptions
- Working directory: /home/b/ResumeBuilder3_bck
- Sandbox: danger-full-access; network enabled; approval_policy=never
- Test gate: python harness.py --run-all-tests must pass before completion

## File Claims
- None (released)

## Decisions
- Intake complete: README.md, docs/TESTING.md, docs/LONG_RUN.md reviewed.
- Use pyupgrade --py312-plus based on runtime Python 3.12.
- Implement optional scanners with non-destructive pyupgrade check and safety API key skip.
- Added codespell/pip-audit/safety/pyupgrade to static analysis and recorded run-all-tests results.
- Enforced missing tools as failures; removed static-analysis timeouts (pip-audit/pytype) and added diagram generation to run-all-tests.
- Hardened static JSON parsing and semgrep invocation; pip-audit spinner disabled to stabilize report stats.
- Updated run_maxcov.sh/run_maxcov_e2e.sh to run --run-all-tests and expanded Dockerfile system deps; install semgrep with --no-deps.
- Added log directory ownership fix in scripts/run_maxcov_e2e.sh to allow summary capture after docker runs.
- New request intake: replace brian_mingus_resume.json with michael_scott_resume.json, add full Michael Scott/David Brent resume data, and keep import safe for real user data.
- Requirements updated with REQ-009 through REQ-013 and docs/FEATURES.json created for the new resume asset work.
- Default assets switched to michael_scott_resume.json; added full resume seed data and safe import guard with --overwrite-resume.
- Test flows updated to emphasize stub DB isolation and documented import workflow.
- New request intake: compile Michael Scott PDF with stubbed DB and perform a vision review.
- New request intake: fix header name wrapping for long names in Typst.

## Open Risks
- Semgrep dependency constraints conflict with reflex; installed via --no-deps in Dockerfile.
- Untracked tool configs created during runs (.pyre_configuration, .semgrep.yml, .pyre/); awaiting direction.
- Resume JSON expansion increases PDF length but run-all-tests passed; monitor UI PDF performance if content grows.

## Checkpoints
- 2025-12-23T07:44:22Z: Optional scanners added; run-all-tests PASS; see docs/TEST_LOG.md.
- 2025-12-23T09:34:46Z: run-all-tests PASS with diagrams; e2e docker run blocked by sudo prompt; see docs/TEST_LOG.md.
- 2025-12-23T10:03:55Z: run-all-tests PASS with JSON parsers fixed; see docs/TEST_LOG.md.
- 2025-12-23T10:22:27Z: run_maxcov_e2e.sh --force PASS with docker run-all-tests; see docs/TEST_LOG.md.
- 2025-12-23T11:46:04Z: REQ-009..REQ-013 complete; run-all-tests PASS; see docs/TEST_LOG.md (2025-12-23T11:44:16Z).
- 2025-12-23T11:54:22Z: REQ-014 complete; PDF compiled and vision review recorded; see docs/TEST_LOG.md (2025-12-23T11:53:38Z).
- 2025-12-23T12:21:30Z: REQ-015 complete; name wrapping fix verified via PDF and run-all-tests PASS; see docs/TEST_LOG.md (2025-12-23T12:20:44Z).

## Reflections
- Resolved: docker e2e run passes after log ownership fix and sudo provided.
- Failed assumption: long resume seed data might break PDF/UI checks.
  - Corrective action: verified run-all-tests with updated Michael Scott resume asset.
  - Prevention: keep seed resume ASCII-only and rerun UI checks after any content expansion.
