#!/usr/bin/env bash
#
# de_batch_test.sh — self-contained batch job for DynamicTREX transfer-correctness testing.
#
# Submit from a login node; the scheduler runs it on a runner machine. Because the runner
# does not inherit the login shell's environment or working directory, this script does its
# own `source` and `cd` — do not rely on anything from the submitting shell.
#
# Submit with SLURM:      sbatch Runnables/scripts/de_batch_test.sh
# Or run directly:        bash   Runnables/scripts/de_batch_test.sh
# (The #SBATCH lines below are plain comments to a bare `bash`, so both work.)
#
# What it runs, on the DE (Germany) dataset:
#   1. simulateAndCompareTransferUpdates          — independent seeded single-batch fuzzing
#   2. a matrix of simulateAndDetailCompareTransferTimeline runs (rate scenario x seed) via
#      timeline_correctness.sh — sequential accumulation timelines, collecting ALL divergences.
#
# Tune the CONFIG block below, or override any variable from the submit line, e.g.:
#   sbatch --export=ALL,SEEDS="1 2 3",ITERATIONS=1000 Runnables/scripts/de_batch_test.sh

#SBATCH --job-name=dtrex-corr
#SBATCH --output=dtrex_corr_%j.log      # combined stdout+stderr, %j = job id
#SBATCH --time=24:00:00                 # walltime cap
#SBATCH --cpus-per-task=32              # transfer discovery is parallel; give it cores
#SBATCH --mem=0                         # 0 = all memory on the node (DE is large)
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

# ============================== CONFIG (override via --export) ==============================
PROJECT_ROOT="${PROJECT_ROOT:-/nfs/home/reuter/DynamicTREX}"
ACTIVATE="${ACTIVATE:-${PROJECT_ROOT}/activate.sh}"
DATASET="${DATASET:-${PROJECT_ROOT}/Datasets/DE/dynamic.binary}"

BUILD="${BUILD:-1}"                     # 1 = run `make DynamicRelease` first, 0 = skip
RUN_ITERATIONS="${RUN_ITERATIONS:-1}"   # 1 = run the independent-iteration fuzzing phase
RUN_TIMELINE="${RUN_TIMELINE:-1}"       # 1 = run the accumulation-timeline matrix phase

# Output location (per-run logs, timing CSVs, summaries).
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/results/de_correctness_$(date +%Y%m%d_%H%M%S)}"

# --- iteration-fuzzing phase parameters ---
ITER_CURRENT_TIME="${ITER_CURRENT_TIME:-28800}"   # 08:00
ITER_BASE_SEED="${ITER_BASE_SEED:-1}"
ITERATIONS="${ITERATIONS:-1000}"
ITER_CANC="${ITER_CANC:-50}"
ITER_DELAYS="${ITER_DELAYS:-1000}"
ITER_SKIPS="${ITER_SKIPS:-100}"

# --- timeline-matrix phase parameters (consumed by timeline_correctness.sh) ---
export SEEDS="${SEEDS:-1 1000 2000}"
export START="${START:-28800}"          # 08:00
export END="${END:-79200}"              # 22:00
export STEP="${STEP:-60}"               # one update batch per simulated minute
export TRANSFERSET="${TRANSFERSET:-both}"
export THREADS="${THREADS:-max}"
export MAX_SKIP="${MAX_SKIP:-2}"
# Fraction of delays that run EARLY / touch a SINGLE stop event. 0 reproduces the previous
# generator; both paths are unreachable with whole-suffix positive delays alone.
export EARLY_SHARE="${EARLY_SHARE:-0}"
export SINGLE_EVENT_SHARE="${SINGLE_EVENT_SHARE:-0}"
# Shared rate parameters for both phases:
export CANC_HORIZON="${CANC_HORIZON:-7200}"
export SKIP_HORIZON="${SKIP_HORIZON:-7200}"
export MIN_DELAY="${MIN_DELAY:-60}"
export MAX_DELAY="${MAX_DELAY:-600}"
# ===========================================================================================

echo "==== DynamicTREX DE correctness batch ===="
echo "Host:        $(hostname)"
echo "Date:        $(date)"
echo "Project:     ${PROJECT_ROOT}"
echo "Dataset:     ${DATASET}"
echo "Output dir:  ${OUTDIR}"
echo

# 1) Environment + working directory (runner does NOT inherit these from the login node).
if [[ ! -f "${ACTIVATE}" ]]; then
    echo "error: activate script not found: ${ACTIVATE}" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${ACTIVATE}"

RUNNABLES="${PROJECT_ROOT}/Runnables"
cd "${RUNNABLES}"

if [[ ! -f "${DATASET}" ]]; then
    echo "error: dataset not found: ${DATASET}" >&2
    exit 2
fi

mkdir -p "${OUTDIR}"

# 2) Build (idempotent; make is a no-op if already current).
if [[ "${BUILD}" == "1" ]]; then
    echo "-- Building DynamicRelease --"
    make DynamicRelease
fi
DYNAMIC_BIN="${RUNNABLES}/Dynamic"
if [[ ! -x "${DYNAMIC_BIN}" ]]; then
    echo "error: Dynamic binary missing after build: ${DYNAMIC_BIN}" >&2
    exit 2
fi

overall_status=0

# 3) Phase 1 — independent-iteration fuzzing. This command already collects every divergence
#    (it never stops early) and rebuilds its pristine baseline in memory per iteration.
if [[ "${RUN_ITERATIONS}" == "1" ]]; then
    echo
    echo "==== Phase 1: simulateAndCompareTransferUpdates (${ITERATIONS} iterations) ===="
    iter_out="${OUTDIR}/iterations.detail.txt"
    iter_timing="${OUTDIR}/iterations.timing.csv"
    iter_log="${OUTDIR}/iterations.log"
    # Parameter order: input curTime seed iters out timingCsv \
    #   canc delays skips cancHoriz skipHoriz minDelay maxDelay maxSkip \
    #   earlyShare singleEventShare transferSet threads
    "${DYNAMIC_BIN}" > "${iter_log}" 2>&1 <<CMDS || true
simulateAndCompareTransferUpdates ${DATASET} ${ITER_CURRENT_TIME} ${ITER_BASE_SEED} ${ITERATIONS} ${iter_out} ${iter_timing} ${ITER_CANC} ${ITER_DELAYS} ${ITER_SKIPS} ${CANC_HORIZON} ${SKIP_HORIZON} ${MIN_DELAY} ${MAX_DELAY} ${MAX_SKIP} ${EARLY_SHARE} ${SINGLE_EVENT_SHARE} ${TRANSFERSET} ${THREADS}
quit
CMDS
    if grep -q "No transfer divergences found" "${iter_log}"; then
        echo "Phase 1: PASSED (no divergences) — see ${iter_log}"
    else
        echo "Phase 1: DIVERGENCE(S) DETECTED — inspect ${iter_out} / ${iter_log}"
        overall_status=1
    fi
fi

# 4) Phase 2 — accumulation-timeline matrix (rate scenario x seed). Uses timeline_correctness.sh,
#    which streams every run into one Dynamic process and now collects ALL divergences per run.
if [[ "${RUN_TIMELINE}" == "1" ]]; then
    echo
    echo "==== Phase 2: timeline matrix (simulateAndDetailCompareTransferTimeline) ===="
    HARNESS="${RUNNABLES}/scripts/timeline_correctness.sh"
    if [[ ! -x "${HARNESS}" ]]; then
        echo "error: timeline harness not found/executable: ${HARNESS}" >&2
        exit 2
    fi
    # Its own env-overridable knobs (SEEDS/START/END/STEP/MAX_SKIP/... ) are already exported above.
    if OUTDIR="${OUTDIR}/timeline" DYNAMIC_BIN="${DYNAMIC_BIN}" "${HARNESS}" "${DATASET}"; then
        echo "Phase 2: all timelines PASSED."
    else
        echo "Phase 2: one or more timelines DID NOT PASS — see ${OUTDIR}/timeline/"
        overall_status=1
    fi
fi

echo
echo "==== Batch finished at $(date). Overall status: ${overall_status} (0 = all passed) ===="
echo "Results under: ${OUTDIR}"
exit "${overall_status}"
