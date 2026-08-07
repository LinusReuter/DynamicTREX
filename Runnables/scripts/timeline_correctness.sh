#!/usr/bin/env bash
#
# timeline_correctness.sh
#
# Drives the `Dynamic` shell's `simulateAndDetailCompareTransferTimeline` command over a
# matrix of {update-rate scenario} x {base seed}. Each timeline builds its transfer store
# once in memory and applies sequential minute-by-minute updates, comparing the incremental
# export against a full rebuild at every step. This is the accumulation-heavy correctness
# path (state carries across steps, like the production once-a-minute pipeline) and it never
# serializes a store to disk, so it scales to large networks.
#
# All commands are streamed into a single `Dynamic` process via stdin, so the binary is
# launched once. Each timeline re-loads the input binary internally (a few seconds) and is
# otherwise independent of the others.
#
# Usage:
#   Runnables/scripts/timeline_correctness.sh <dynamic.binary> [partition_file]
#
# Environment overrides (all optional):
#   DYNAMIC_BIN   path to the Dynamic executable            (default: ./Runnables/Dynamic)
#   OUTDIR        directory for per-run logs + timing CSVs   (default: ./timeline_runs)
#   SEEDS         space-separated base seeds                 (default: "1 1000 2000")
#   START END STEP  timeline window in seconds               (default: 0 86400 60)
#   TRANSFERSET   full | reduced | both                      (default: both)
#   THREADS       worker threads or "max"                    (default: max)
#   CANC_HORIZON SKIP_HORIZON MIN_DELAY MAX_DELAY MAX_SKIP    (defaults: 7200 7200 60 600 1)
#   EARLY_SHARE SINGLE_EVENT_SHARE  fraction of delays that run early / touch one stop (default: 0 0)
#   SCENARIOS     newline-separated "name canc delays skips" rows (see defaults below)
#
# Exit status: 0 if every run reported success, 1 if any run diverged or errored.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_RUNNABLES="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT="${1:-}"
PARTITION="${2:-}"
if [[ -z "${INPUT}" ]]; then
    echo "usage: $0 <dynamic.binary> [partition_file]" >&2
    exit 2
fi
if [[ ! -f "${INPUT}" ]]; then
    echo "error: input binary not found: ${INPUT}" >&2
    exit 2
fi

DYNAMIC_BIN="${DYNAMIC_BIN:-${REPO_RUNNABLES}/Dynamic}"
if [[ ! -x "${DYNAMIC_BIN}" ]]; then
    echo "error: Dynamic executable not found/executable: ${DYNAMIC_BIN}" >&2
    echo "       build it first:  (cd ${REPO_RUNNABLES} && make DynamicRelease)" >&2
    exit 2
fi

OUTDIR="${OUTDIR:-./timeline_runs}"
SEEDS="${SEEDS:-1 1000 2000}"
START="${START:-0}"
END="${END:-86400}"
STEP="${STEP:-60}"
TRANSFERSET="${TRANSFERSET:-both}"
THREADS="${THREADS:-max}"
CANC_HORIZON="${CANC_HORIZON:-7200}"
SKIP_HORIZON="${SKIP_HORIZON:-7200}"
MIN_DELAY="${MIN_DELAY:-60}"
MAX_DELAY="${MAX_DELAY:-600}"
MAX_SKIP="${MAX_SKIP:-1}"
# Fraction of delays that run EARLY, and fraction applied to a SINGLE stop event. Both default
# to 0, which reproduces the pre-existing generator; raise them to exercise the fall-through and
# index-neighbour paths that whole-suffix positive delays can never reach.
EARLY_SHARE="${EARLY_SHARE:-0}"
SINGLE_EVENT_SHARE="${SINGLE_EVENT_SHARE:-0}"

# Scenario rows: "name  expected_cancellations  expected_delays  expected_skipped_trips".
# The isolating scenarios (only one update kind active) make it obvious which update path a
# divergence came from; the mixed rows exercise interactions.
SCENARIOS="${SCENARIOS:-$(cat <<'EOF'
cancellations_only 200   0   0
delays_only          0 500   0
skips_only           0   0 200
moderate_mixed     100 100 100
heavy_mixed        500 500 500
realistic           20 300  30
EOF
)}"

mkdir -p "${OUTDIR}"
MASTER_LOG="${OUTDIR}/master.log"
: > "${MASTER_LOG}"

# --- Build the stdin command stream for a single Dynamic session -------------------------
CMD_STREAM="$(mktemp "${TMPDIR:-/tmp}/timeline_cmds.XXXXXX")"
trap 'rm -f "${CMD_STREAM}"' EXIT

if [[ -n "${PARTITION}" ]]; then
    if [[ ! -f "${PARTITION}" ]]; then
        echo "error: partition file not found: ${PARTITION}" >&2
        exit 2
    fi
    # Applied once, up front, to the in-memory data used by every subsequent timeline.
    echo "loadAndApplyDynamicPartition ${PARTITION} ${INPUT}" >> "${CMD_STREAM}"
fi

declare -a RUN_TAGS=()
declare -a RUN_OUTS=()

while read -r name canc delays skips; do
    [[ -z "${name:-}" ]] && continue
    for seed in ${SEEDS}; do
        tag="${name}_seed${seed}"
        out="${OUTDIR}/${tag}.detail.txt"
        timing="${OUTDIR}/${tag}.timing.csv"
        RUN_TAGS+=("${tag}")
        RUN_OUTS+=("${out}")
        # Positional parameter order must match SimulateAndDetailCompareTransferTimeline:
        #   input start end step seed out timingCsv \
        #   canc delays skips cancHorizon skipHorizon minDelay maxDelay maxSkip \
        #   earlyShare singleEventShare transferSet threads
        printf 'simulateAndDetailCompareTransferTimeline %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
            "${INPUT}" "${START}" "${END}" "${STEP}" "${seed}" "${out}" "${timing}" \
            "${canc}" "${delays}" "${skips}" \
            "${CANC_HORIZON}" "${SKIP_HORIZON}" "${MIN_DELAY}" "${MAX_DELAY}" "${MAX_SKIP}" \
            "${EARLY_SHARE}" "${SINGLE_EVENT_SHARE}" \
            "${TRANSFERSET}" "${THREADS}" >> "${CMD_STREAM}"
    done
done <<< "${SCENARIOS}"

echo "quit" >> "${CMD_STREAM}"

RUN_COUNT="${#RUN_TAGS[@]}"
echo "Running ${RUN_COUNT} timeline(s) over window [${START}, ${END}) step ${STEP}s, transferSet=${TRANSFERSET}."
echo "Streaming commands into: ${DYNAMIC_BIN}"
echo "Per-run logs + timing CSVs under: ${OUTDIR}"
echo

# --- Run one Dynamic session, tee everything to the master log ---------------------------
"${DYNAMIC_BIN}" < "${CMD_STREAM}" 2>&1 | tee "${MASTER_LOG}"

# --- Summarize ---------------------------------------------------------------------------
echo
echo "================ TIMELINE CORRECTNESS SUMMARY ================"
failures=0
for i in "${!RUN_TAGS[@]}"; do
    tag="${RUN_TAGS[$i]}"
    out="${RUN_OUTS[$i]}"
    status="UNKNOWN"
    if [[ -f "${out}" ]] && grep -q "FAILED" "${out}"; then
        status="DIVERGED"
    elif [[ -f "${out}" ]] && grep -q "Status: Success" "${out}"; then
        status="PASSED"
    fi
    if [[ "${status}" != "PASSED" ]]; then
        failures=$((failures + 1))
    fi
    printf '  [%-8s] %s\n' "${status}" "${tag}"
done

echo "-------------------------------------------------------------"
if [[ "${failures}" -eq 0 ]]; then
    echo "All ${RUN_COUNT} timeline(s) PASSED."
    exit 0
else
    echo "${failures}/${RUN_COUNT} timeline(s) DID NOT PASS — inspect the per-run *.detail.txt in ${OUTDIR}."
    exit 1
fi
