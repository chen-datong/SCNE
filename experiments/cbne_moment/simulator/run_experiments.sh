#!/bin/bash

# 固定参数

export OPENBLAS_MAIN_FREE=1
export OMP_PROC_BIND=false
export OMP_PLACES=threads
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3

MAX_DIM=500
S=1000
H=1.0
P=0.20

# 扫描的 depth 列表（你可以自己加）
DEPTHS=(35)

# 扫描的 sampleNum 列表
NS=(8)

# 每个条件重复次数
REPEATS=1000

# 输出目录
OUTDIR="estimate_third"
mkdir -p $OUTDIR

echo "Starting experiments..."
echo ""


for DEPTH in "${DEPTHS[@]}"
do
    echo "=== Running depth = $DEPTH ==="

    for N in "${NS[@]}"
    do
        echo "  N = $N"
        exp_id=0
        for ((i=1; i<=REPEATS; i++))
        do
            ./main $N $DEPTH $S $MAX_DIM $P $H $exp_id 
            ((exp_id++))

            if (( i % 500 == 0 )); then
                echo "    progress: $i / $REPEATS"
            fi
        done
    done
done

echo ""
echo "ALL EXPERIMENTS FINISHED!"
echo "Total experiments = $exp_id"
