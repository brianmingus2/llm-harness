# KNOWLEDGE_BASE

## Skills
- Run full validation: `python harness.py --run-all-tests` (includes maxcov, clean reflex start, UI check).
- Generate diagrams: `python scripts/generate_diagrams.py` (writes to diagrams/).

## Lessons
- Run pyupgrade on temporary copies to avoid mutating tracked files during scans.
- [deps][REQ-006] Semgrep conflicts with reflex click/rich constraints; install semgrep with `pip install --no-deps semgrep==1.146.0` and keep click/rich for reflex.
- [assets][REQ-011] Asset imports refuse to overwrite existing resumes unless `--overwrite-resume` is set; run-all-tests uses stub DB to avoid touching user data.
