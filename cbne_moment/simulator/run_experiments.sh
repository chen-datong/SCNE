#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export OPENBLAS_MAIN_FREE=1
export OMP_PROC_BIND=false
export OMP_PLACES=threads
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3

max_dim() {
    case "$1" in
        17) printf '600' ;;
        20) printf '900' ;;
        *) printf '500' ;;
    esac
}
S=100000000
H=1.0
P=0.20
NS=(5 8 11 14 17 20)
REPEATS=${REPEATS:-5000}
OUTDIR="../../data/cbne_moment/TIM_complete"

MODE=${1:-run}
case "$MODE" in
    --dry-run|--prepare-states|run) ;;
    *) printf 'Usage: bash run_experiments.sh [--dry-run|--prepare-states]\n' >&2; exit 2 ;;
esac
if [[ ! "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'REPEATS must be a positive integer\n' >&2
    exit 2
fi
if [[ "$MODE" == "--dry-run" ]]; then
    printf 'Total N_M=%s; repeats per n=%s\n' "$S" "$REPEATS"
    if [[ ! -x ./main ]]; then printf 'Missing executable: ./main (build with make first)\n'; fi
    for N in "${NS[@]}"; do
        if (( N >= 17 )); then DEPTH=50; else DEPTH=35; fi
        STATE="MPS/ising_N${N}_J1.00_h1.00.mps"
        printf 'n=%s depth=%s MaxDim=%s state=%s' "$N" "$DEPTH" "$(max_dim "$N")" "$STATE"
        if [[ -f "$STATE" ]]; then printf ' [present]\n'; else printf ' [missing]\n'; fi
    done
    exit 0
fi
if [[ ! -x ./main ]]; then
    printf 'Missing ./main. Compile with make LIBRARY_DIR=/path/to/itensor first.\n' >&2
    exit 1
fi
if [[ "$MODE" == "--prepare-states" ]]; then
    mkdir -p MPS
    for N in "${NS[@]}"; do
        ./main --prepare-state "$N" "$H"
    done
    exit 0
fi

# Check the complete input grid before starting measurements.
for N in "${NS[@]}"; do
    STATE="MPS/ising_N${N}_J1.00_h1.00.mps"
    if [[ ! -f "$STATE" ]]; then
        printf 'Missing %s. Run bash run_experiments.sh --prepare-states first.\n' "$STATE" >&2
        exit 1
    fi
    if (( N >= 17 )); then DEPTH=50; else DEPTH=35; fi
    FILE="$OUTDIR/n${N}_depth${DEPTH}_J1.00_h1.00_p0.20_Sample${S}.jsonl"
    if [[ -e "$FILE" ]]; then
        printf 'Output exists: %s. Preserve or inspect it before a fresh run.\n' "$FILE" >&2
        exit 1
    fi
done
mkdir -p "$OUTDIR"
for N in "${NS[@]}"; do
    if (( N >= 17 )); then DEPTH=50; else DEPTH=35; fi
    MAX_DIM=$(max_dim "$N")
    printf 'n=%s depth=%s MaxDim=%s total N_M=%s repeats=%s\n' "$N" "$DEPTH" "$MAX_DIM" "$S" "$REPEATS"
    for ((i=1; i<=REPEATS; i++)); do
        EXP_ID=$((i-1))
        ./main "$N" "$DEPTH" "$S" "$MAX_DIM" "$P" "$H" "$EXP_ID" "$OUTDIR"
        if (( i % 500 == 0 )); then printf 'progress: %s/%s\n' "$i" "$REPEATS"; fi
    done
done
printf 'Finished: %s trials\n' "$((REPEATS * ${#NS[@]}))"
