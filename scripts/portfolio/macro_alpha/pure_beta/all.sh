#!/usr/bin/env bash
#
# Run every Pure Beta (S_I \\ S_S) job: all sentiment backends x both target
# distributions. Extra arguments are forwarded to each backend script.
#
# Examples:
#   ./scripts/portfolio/macro_alpha/pure_beta/all.sh
#   ./scripts/portfolio/macro_alpha/pure_beta/all.sh --holdings 40 60
#   BACKENDS="gpt4o_mini finbert" DISTRIBUTIONS=gaussian ./scripts/portfolio/macro_alpha/pure_beta/all.sh
#   BETA_THRESHOLD=0.3 ./scripts/portfolio/macro_alpha/pure_beta/all.sh --holdings 40

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BACKENDS="${BACKENDS:-finbert gpt4o_mini llama2 mistral}"
DISTRIBUTIONS="${DISTRIBUTIONS:-gaussian student_t}"

failed=()
for dist in $DISTRIBUTIONS; do
  for backend in $BACKENDS; do
    script="./$dist/$backend.sh"
    if [[ ! -x "$script" ]]; then
      echo "skipping $script (not found or not executable)" >&2
      failed+=("$dist/$backend: missing")
      continue
    fi

    echo "=== pure_beta / $dist / $backend ==="
    if ! "$script" "$@"; then
      echo "FAILED: pure_beta/$dist/$backend" >&2
      failed+=("$dist/$backend")
    fi
  done
done

if ((${#failed[@]})); then
  printf 'failed runs: %s\n' "${failed[*]}" >&2
  exit 1
fi

echo "all pure_beta runs complete -> benchmark/pure_beta/"
