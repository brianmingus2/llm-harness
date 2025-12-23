#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_maxcov_e2e.sh --force

Runs a full Docker reset, builds the maxcov image, executes maximum coverage,
executes run-all-tests, captures the coverage summary, then resets Docker again.

WARNING: This deletes ALL Docker containers, images, volumes, and networks
on this machine.
EOF
}

force=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force|-y)
      force=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$force" -ne 1 ]]; then
  usage >&2
  exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
cd "${root_dir}"

log_file="${root_dir}/maxcov_logs/maximum_coverage.log"
summary_file="${root_dir}/maxcov_logs/maximum_coverage.summary.txt"

ensure_log_dir() {
  ${SUDO} mkdir -p "${root_dir}/maxcov_logs"
  if [[ -n "${SUDO}" ]]; then
    ${SUDO} chown -R "$(id -u):$(id -g)" "${root_dir}/maxcov_logs"
  fi
}

nuke_docker() {
  local ids=""
  ids="$(${SUDO} docker ps -aq || true)"
  if [[ -n "${ids}" ]]; then
    echo "${ids}" | xargs ${SUDO} docker rm -f
  fi

  ids="$(${SUDO} docker images -aq || true)"
  if [[ -n "${ids}" ]]; then
    echo "${ids}" | xargs ${SUDO} docker rmi -f
  fi

  ids="$(${SUDO} docker volume ls -q || true)"
  if [[ -n "${ids}" ]]; then
    echo "${ids}" | xargs ${SUDO} docker volume rm -f
  fi

  ids="$(${SUDO} docker network ls --format '{{.Name}}' 2>/dev/null | grep -v -E '^(bridge|host|none)$' || true)"
  if [[ -n "${ids}" ]]; then
    echo "${ids}" | xargs ${SUDO} docker network rm -f
  fi

  ${SUDO} docker system prune -af --volumes || true
}

ensure_log_dir

start_ts="$(date -u +%s)"
echo "[e2e] docker reset (pre)"
nuke_docker

echo "[e2e] build maxcov image"
${SUDO} docker compose build --no-cache maxcov

echo "[e2e] run all tests"
${SUDO} docker compose run --rm -T maxcov

echo "[e2e] capture coverage summary"
ensure_log_dir
summary_line="$(grep -a "Coverage (harness.py)" "${log_file}" | tail -n 1 || true)"
end_ts="$(date -u +%s)"
duration_s=$(( end_ts - start_ts ))
duration_h=$(( duration_s / 3600 ))
duration_m=$(( (duration_s % 3600) / 60 ))
duration_rem_s=$(( duration_s % 60 ))
{
  echo "Start (UTC): $(date -u -d "@${start_ts}" +%Y-%m-%dT%H:%M:%SZ)"
  echo "End (UTC): $(date -u -d "@${end_ts}" +%Y-%m-%dT%H:%M:%SZ)"
  printf "Duration: %dh %dm %ds\n" "${duration_h}" "${duration_m}" "${duration_rem_s}"
  echo "Timestamp (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -n "${summary_line}" ]]; then
    echo "${summary_line}"
  else
    echo "Coverage summary not found."
  fi
} > "${summary_file}"
echo "[e2e] ${summary_line:-Coverage summary not found.}"
echo "[e2e] summary written to ${summary_file}"

echo "[e2e] docker reset (post)"
nuke_docker
