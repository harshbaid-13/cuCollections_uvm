#!/bin/bash
set -euo pipefail

# ----------------------------
# Configuration
# ----------------------------
EXECUTABLE=./a.out
TRACE_DIR=/data/heterods-trace
RESULTS_DIR=Results/Inserts

operations=(
    # 2000000000
    # 2500000000
    # 3000000000
    3500000000
    4000000000
)

files=(
    # "insert_trace-400e7-100-add-20-dup-DENSE_REPEAT.bin"
    # "insert_trace-400e7-100-add-no-dup-MONOTONIC_INCREASE.bin"
    # "insert_trace-400e7-100-add-20-dup-SPARSE_REPEAT.bin"
    # "insert_trace-400e7-100-add-no-dup-DENSE_UNIQUE_1e9_Clusters.bin"
    # "insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE_5e8_Random_Clusters.bin"
    # "insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE_ZigZag.bin"
    # "insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin"
    "insert_trace-400e7-100-add-no-dup-DENSE_UNIQUE.bin"
)

# ----------------------------
# Build
# ----------------------------
mkdir -p "${RESULTS_DIR}"

nvcc -std=c++17 \
     --expt-extended-lambda \
     -arch=sm_75 \
     insert_or_assign_test.cu \
     -I../. \
     -I../../include \
     -I/data/srinjoy/cccl/thrust \
     -I/data/srinjoy/cccl/libcudacxx/include \
     -I/data/srinjoy/cccl/cub \
     -I/data/srinjoy/catch2/src/ \
     -I/data/srinjoy/catch2/build_cmake/generated-includes \
     -L/data/srinjoy/catch2/build_cmake/src \
     -lCatch2 \
     -lCatch2Main

# ----------------------------
# Benchmark loop
# ----------------------------
for file in "${files[@]}"; do

    case "$file" in
        insert_trace-400e7-100-add-20-dup-DENSE_REPEAT.bin)
            folder="${RESULTS_DIR}/Dense_Repeat"
            ;;
        insert_trace-400e7-100-add-no-dup-DENSE_UNIQUE.bin)
            folder="${RESULTS_DIR}/Dense_Unique"
            ;;
        insert_trace-400e7-100-add-no-dup-MONOTONIC_INCREASE.bin)
            folder="${RESULTS_DIR}/Monotonic_Increase"
            ;;
        insert_trace-400e7-100-add-no-dup-DENSE_UNIQUE_1e9_Clusters.bin)
            folder="${RESULTS_DIR}/1e9_Clusters"
            ;;
        insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE_5e8_Random_Clusters.bin)
            folder="${RESULTS_DIR}/Random_Clusters"
            ;;
        insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE_ZigZag.bin)
            folder="${RESULTS_DIR}/ZigZag"
            ;;
        insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin)
        folder="${RESULTS_DIR}/Sparse_Unique"
            ;;
        *)
            folder="${RESULTS_DIR}/Sparse_Repeat"
            ;;
    esac

    mkdir -p "${folder}"

    insert_path="${TRACE_DIR}/${file}"
    # find_path="${TRACE_DIR}/${file}"

    echo "=============================================="
    echo "Trace file : ${file}"
    echo "Output dir : ${folder}"
    echo "=============================================="

    for ops in "${operations[@]}"; do
        echo "  → Running num-keys=${ops}"

        "${EXECUTABLE}" \
            --insert-path "${insert_path}" \
            --num-keys "${ops}" \
            > "${folder}/${ops}.log" 2>&1
    done
done
