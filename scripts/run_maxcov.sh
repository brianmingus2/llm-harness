#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/var/log/maxcov"
LOG_FILE="${LOG_DIR}/maximum_coverage.log"

mkdir -p "${LOG_DIR}"

export MAX_COVERAGE_CONTAINER="1"
export MAX_COVERAGE_LOG="1"
export MAX_COVERAGE_STUB_DB="1"

python harness.py --run-all-tests 2>&1 | tee "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
