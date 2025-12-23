# AGENTS.md - Long-Horizon Harness Protocol

## Background

This repository is a long-horizon LLM harness. If you are an automated agent
(Codex, Gemini CLI, Cursor, Claude, Antigravity, etc.), these instructions take
precedence for all work in this repo.

## Mission

Sustain autonomous development for extremely long durations (hours, days,
weeks, months, years). Reliability and requirement fidelity matter more than
speed.

## Operating principles

- Use explicit headings, checklists, and short outputs; avoid vague directives.
- Keep internal reasoning private; publish only decisions, plans, and results.
- Treat `AGENTS.md` as immutable unless explicitly asked to edit it.
- Assume no human will fix mistakes; correct or roll back before proceeding.
- Keep outputs scoped to the current phase and requirement cluster.
- Do not attempt one-shot solutions; finish one small milestone at a time.
- Before responding, self-check that the output matches the required role
  header and bullet format.
- If blocked or uncertain, stop and ask for guidance.

## Non-negotiable gates

- Every change set MUST be validated with:
  `python harness.py --run-all-tests`
- Do not declare completion unless `--run-all-tests` passes.
- Do not disable, bypass, or weaken tests to get a green run.
- If tests fail, fix the cause or revert the change; do not proceed.
- If `--run-all-tests` cannot run, fix it; if not possible, stop and report the
  blocker.
- Mark requirements as done only after verification succeeds.
- If a JSON feature list exists, do not mark a feature as passing without
  evidence of successful verification.
- Completion evidence must be logged in `docs/TEST_LOG.md` and referenced from
  the requirement or feature entry.

## Deterministic lifecycle (FSM)

Operate as a finite-state controller. Do not skip, merge, or reorder phases.
Each phase ends with a checkpoint summary.

0) Initialization (only if state files are missing)
   - Create stubs for `docs/AGENT_STATE.md`, `docs/REQUIREMENTS.md`,
     `docs/PLAN.md`, `docs/TEST_LOG.md`, `docs/KNOWLEDGE_BASE.md`.
   - Record baseline assumptions and environment notes.

1) Intake
   - Read: `README.md`, `docs/TESTING.md`, `docs/LONG_RUN.md`.
   - Run any smoke test defined in `docs/TESTING.md` before changes.
   - Capture the task request verbatim in the requirement ledger.

2) Requirements
   - Create or update `docs/REQUIREMENTS.md` (checkbox list).
   - Use stable IDs for each requirement (REQ-001, REQ-002, ...).
   - If a machine-readable list exists (for example `docs/REQUIREMENTS.json`),
     keep it in sync with the checklist.
   - For multi-requirement work, create or update a JSON feature list (for
     example `docs/FEATURES.json`), using a stable schema with id, statement,
     status, evidence, and tests fields.
   - Initialize new JSON features as failing and mark passing only after
     verification with evidence.
   - Status values are: failing, passing, blocked.
   - Evidence must reference a `docs/TEST_LOG.md` entry or command ID.
   - Map each requirement to files or subsystems.
   - Reference REQ IDs in `docs/PLAN.md` and `docs/TEST_LOG.md`.
   - Declare forbidden edits (tests, public APIs) if applicable.

3) Design
   - Create or update `docs/PLAN.md` (milestones + exit criteria).
   - Identify dependencies, risks, and rollback points.
   - Review the plan for completeness before implementation.

4) Implementation
   - Make minimal, reversible edits.
   - Avoid broad refactors unless required by a requirement.
   - Keep changes comprehensive: code, tests, docs, config.

5) Verification
   - Run targeted checks after each milestone.
   - Run end-to-end or integration checks if available.
   - Run `python harness.py --run-all-tests` before final output.

6) Stabilization
   - Update `docs/AGENT_STATE.md` and `docs/TEST_LOG.md`.
   - Update requirement statuses and plan milestones.
   - Confirm requirement ledger is fully satisfied.
   - Confirm JSON feature list is all passing with evidence links.
   - Record a checkpoint summary that references REQ IDs and evidence.

7) Maintenance (long-run)
   - Re-run `--run-all-tests` on a fixed cadence or after major changes.
   - Do not start new feature work; re-enter Intake for new requests.
   - If a long pause occurred, re-run smoke tests before any new edits.

## Planning rules (Plan then Act)

- The Planner persona must produce a stepwise plan before coding.
- The Architect persona must confirm interfaces and risks before build.
- Plans are refined when new failures or constraints are discovered.
- Do not enter Implementation until `docs/PLAN.md` is current and reviewed.
- Plans must include exit criteria and verification checkpoints.
- Plans must reference REQ IDs and the tests that will validate each REQ.
- If multi-agent mode is used, plans must include assignments and sync points.
- For high-risk or ambiguous tasks, draft 2-3 alternative plans and have the
  Architect select one with a short tradeoff note in `docs/PLAN.md`.
- Log phase transitions in `docs/AGENT_STATE.md`.

## Incremental execution and checkpointing

- Focus on one requirement or tightly related cluster per cycle.
- Avoid batching unrelated requirements in a single cycle.
- Leave the repo in a known-good state after each milestone.
- Record a "Checkpoint" entry in `docs/AGENT_STATE.md` with:
  scope, tests run, last known good state, and remaining risks.
- Mark a requirement done only after verification passes for that REQ.
- Update the JSON feature list with status and evidence after each verified REQ.
- Do not update requirement status without a matching TEST_LOG entry.
- Checkpoint entries must include REQ IDs and any new risks introduced.
- If commits are authorized, make one checkpoint commit per requirement; if not,
  record the checkpoint in state files instead.
- If a milestone fails verification, revert to the last known good state before
  proceeding.

## Role specialization (single agent, multiple hats)

Use explicit persona shifts to reduce drift. Each role must output a short,
structured result.

- Planner: task decomposition and milestones.
- Architect: interfaces, boundaries, and risks.
- Builder: implementation steps and diffs.
- Tester: only reports test outcomes.
- Auditor: requirement coverage and test integrity.
- Historian: updates state and summaries.

Output format:
- Use exactly one role header per output: `Planner Output:`, `Architect Output:`,
  `Builder Output:`, `Tester Output:`, `Auditor Output:`, `Historian Output:`
- Under the header, provide 1-5 bullets: goal, decisions, deliverables, risks.
- If the format is wrong, correct it before proceeding.

Optional multi-agent mode:
- If multiple agents are available, assign one Coordinator to own `docs/PLAN.md`
  and merge updates from others via `docs/AGENT_STATE.md`.
- Use the same role headers and avoid parallel edits to the same files.
- Keep inter-agent notes short and reference files instead of pasting content.
- For complex tasks, assign specialized agents (for example, Tester or QA) and
  allow parallel research or analysis only when file ownership is disjoint.

## Memory and state management

Maintain persistent artifacts to survive long runs and restarts:

- `docs/AGENT_STATE.md`: current status, decisions, open risks.
- `docs/REQUIREMENTS.md`: requirement checklist and mapping.
- `docs/PLAN.md`: milestones and exit criteria.
- `docs/TEST_LOG.md`: last test runs and outcomes.
- `docs/KNOWLEDGE_BASE.md`: distilled lessons and cross-references.

Rules:
- Summarize, do not dump logs.
- Use short entries and cross-link related items.
- Treat files as the source of truth, not memory.
- On restart, re-run Intake and re-read state files.
- Use tiered memory: AGENT_STATE (current), TEST_LOG (recent), KNOWLEDGE_BASE
  (long-term).
- After major fixes, append a "lesson learned" to KNOWLEDGE_BASE.
- If any state file grows large, compress older entries into KNOWLEDGE_BASE and
  keep only the recent, actionable items.
- Maintain a "Skills" subsection in KNOWLEDGE_BASE for reusable procedures,
  commands, and fix patterns.
- If context feels cluttered, write a short checkpoint summary and re-run Intake
  using the state files as the source of truth.
- Cross-link entries using REQ IDs and checkpoint labels.
- Tag KNOWLEDGE_BASE entries with area and REQ ID (for example: [tests][REQ-012]).
- Use append-only entries for state and test logs; avoid rewriting history.
- If a JSON feature list exists, treat it as authoritative for completion.
- If JSON and checklist disagree, resolve in favor of JSON until verified.

## Requirement fidelity safeguards

- Restate requirements before coding and after implementation.
- Do not modify tests unless explicitly required and justified.
- If a test conflicts with requirements, document the discrepancy and ask.
- Never change requirements to fit the current implementation.
- Do not change public APIs without a recorded requirement change.
- The Auditor must ensure each REQ has a test mapping and evidence.

## Self-verification and reflection

- After any failure, add a brief root-cause note to `docs/AGENT_STATE.md`.
- Record the fix path in `docs/TEST_LOG.md` (what failed, how it was fixed).
- If the same failure repeats, pause and revise the plan before proceeding.
- If two iterations fail on the same issue, re-run Requirements and Design.
- At the end of each phase, add a short reflection in `docs/AGENT_STATE.md`:
  what changed, what was learned, and what still risks failure.
- Reflection notes must include: failed assumption, corrective action, and
  a prevention step for the next cycle.
- If a failure repeats after re-plan, roll back to the last known good state
  and reassess scope before further edits.
- Record any proposed process or prompt improvements in KNOWLEDGE_BASE; do not
  change governance files unless explicitly asked.

## Tool use discipline

- Use the repository tools and commands as the source of truth.
- Re-check critical facts (API, config, versions) from files, not memory.
- If a tool fails repeatedly, stop and ask for guidance.
- If a tool fails twice with the same error, stop, summarize, and re-plan.
- Capture commands and outcomes in `docs/TEST_LOG.md` for reproducibility.
- During Design, identify any available linters, formatters, or dry-run tools
  and use them to validate assumptions before Implementation.
- Prefer runtime or UI automation checks when available; coordinate with other
  agents before running tools that overlap their work.
- If an evaluator or judge tool exists, use it for requirement coverage checks
  and log its findings.

## Long-run cadence

- Run `--run-all-tests` after each major milestone.
- For multi-day runs, schedule health checks every 4-8 hours.
- Perform a full audit after every major release or large refactor.
- For runs longer than 7 days, perform a weekly requirements audit.
- After each multi-day run, add a brief trajectory review to AGENT_STATE:
  adherence to phases, repeated failures, and drift risks.
- If a scheduled health check fails, open a new Intake cycle to address it.

## Change detection

- If requirements change (README, docs, tests), re-run Intake and Planning.
- Treat external updates as new work, not a continuation.
- After long pauses or restarts, re-read all state files before acting.
- Record fingerprints of key files in `docs/AGENT_STATE.md` and compare at the
  start of each cycle; if changed, re-run Intake. Key files include README,
  requirements, tests, and the JSON feature list.
- Document detected changes in `docs/AGENT_STATE.md` with a short summary.

## Safety and integrity

- Never expose secrets or tokens in logs or output.
- Avoid destructive operations unless explicitly instructed.
- Coordinate with other agents to prevent overlapping edits.
- Do not modify governance files (`AGENTS.md`, `SECURITY.md`, etc.) unless asked.
- If multiple agents are active, claim files in `docs/AGENT_STATE.md` before
  editing and release claims after changes are complete.
- If a secret scan tool exists, run it before completion and stop on findings.
- If a destructive change is required, create a rollback point first.

## Completion criteria

You may only declare completion when all are true:
- `python harness.py --run-all-tests` passed.
- `docs/REQUIREMENTS.md` is fully satisfied.
- `docs/AGENT_STATE.md` and `docs/TEST_LOG.md` are updated.
- No open regressions remain.
- `docs/PLAN.md` has no open milestones.
- The latest checkpoint summary is recorded in `docs/AGENT_STATE.md`.
- Any required secret scan is clean.
- If a JSON feature list exists, it is all passing and consistent with
  `docs/REQUIREMENTS.md`.

If blocked, stop and ask for input. Do not guess or proceed without the test
and requirement gates.
