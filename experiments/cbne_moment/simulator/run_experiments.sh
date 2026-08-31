#!/bin/bash

# 固定参数

export OPENBLAS_MAIN_FREE=1
export OMP_PROC_BIND=false
export OMP_PLACES=threads
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3

MAX_DIM=500
S=100000000
H=1.0
P=0.20

# 论文 SE_vs_NM_and_n.pdf 右图：所有系统大小固定 N_M=10^8。
NS=(5 8 11 14 17 20)

# 每个条件重复次数
REPEATS=5000

# 输出目录
OUTDIR="../data/TIM_complete"
mkdir -p $OUTDIR

echo "Starting experiments..."
echo ""


for N in "${NS[@]}"
do
    if (( N >= 17 )); then DEPTH=50; else DEPTH=35; fi
    echo "=== Running N = $N, depth = $DEPTH, N_M = $S ==="
    exp_id=0
    for ((i=1; i<=REPEATS; i++))
    do
        ./main $N $DEPTH $S $MAX_DIM $P $H $exp_id "$OUTDIR"
        ((exp_id++))

        if (( i % 500 == 0 )); then
            echo "    progress: $i / $REPEATS"
        fi
    done
done

echo ""
echo "ALL EXPERIMENTS FINISHED!"
echo "Total experiments = $((REPEATS * ${#NS[@]}))"
