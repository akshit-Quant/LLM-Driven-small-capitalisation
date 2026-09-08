#!/usr/bin/env bash
#
# Run every macro alpha-beta job: all three strategies x all sentiment backends x
# both target distributions. Extra arguments are forwarded to each script.
#
# This is the full grid — 3 strategies x 4 backends x 2 distributions x 8 holding
# periods of model training. Start with a single strategy and holding period.
#
# Examples:
#   ./scripts/portfolio/macro_alpha/all.sh --holding 40
#   STRATEGIES=pure_alpha ./scripts/portfolio/macro_alpha/all.sh
#   STRATEGIES="pure_alpha pure_beta" BACKENDS=finbert ./scripts/portfolio/macro_alpha/all.sh --holdings 40 60

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

STRATEGIES="${STRATEGIES:-pure_alpha pure_beta beta}"

failed=()
for strat in $STRATEGIES; do
  script="./$strat/all.sh"
  if [[ ! -x "$script" ]]; then
    echo "skipping $script (not found or not executable)" >&2
    failed+=("$strat: missing")
    continue
  fi

  echo "########## strategy: $strat ##########"
  if ! "$script" "$@"; then
    echo "FAILED: $strat" >&2
    failed+=("$strat")
  fi
done

if ((${#failed[@]})); then
  printf 'failed strategies: %s\n' "${failed[*]}" >&2
  exit 1
fi

echo "all macro alpha-beta runs complete -> benchmark/"
