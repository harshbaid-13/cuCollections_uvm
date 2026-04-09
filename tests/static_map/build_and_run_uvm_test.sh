#!/usr/bin/env bash
set -euo pipefail

# Build and run uvm_single_gpu_test.cu (CUDA 13.2 + CCCL).
# Usage: ./build_and_run_uvm_test.sh [N]
#   N defaults to 5000000 (passed as first argument to ./uvm_test).

cd "$(dirname "${BASH_SOURCE[0]}")"

nvcc -std=c++17 \
     --expt-extended-lambda \
     -arch=sm_86 \
     -I../. \
     -I../../include \
     -I/usr/local/cuda-13.2/targets/x86_64-linux/include/cccl \
     uvm_single_gpu_test.cu \
     -o uvm_test

N="${1:-5000000}"
./uvm_test "${N}"
