#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace cuco {
namespace detail {
namespace multi_gpu {

__host__ __device__ inline uint32_t partition_hash(int32_t k) {
  uint32_t x = static_cast<uint32_t>(k);
  x ^= x >> 16; x *= 0x85ebca6bU; x ^= x >> 13; x *= 0xc2b2ae35U; x ^= x >> 16;
  return x;
}

__host__ __device__ inline uint32_t partition_hash(int64_t k) {
  uint64_t x = static_cast<uint64_t>(k);
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
  return static_cast<uint32_t>(x);
}

template <typename Key>
__host__ __device__ inline int partition_id(Key const& key, int num_gpus) {
  return static_cast<int>(partition_hash(key) % static_cast<uint32_t>(num_gpus));
}

template <typename Pair>
__global__ void count_per_partition_kernel(Pair const* pairs, std::size_t n, int num_gpus, std::size_t* counts) {
  auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = tid; i < n; i += stride) {
    int pid = partition_id(pairs[i].first, num_gpus);
    atomicAdd(reinterpret_cast<unsigned long long*>(&counts[pid]), 1ULL);
  }
}

template <typename Key>
__global__ void count_keys_per_partition_kernel(Key const* keys, std::size_t n, int num_gpus, std::size_t* counts) {
  auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = tid; i < n; i += stride) {
    int pid = partition_id(keys[i], num_gpus);
    atomicAdd(reinterpret_cast<unsigned long long*>(&counts[pid]), 1ULL);
  }
}

template <typename Pair>
__global__ void scatter_pairs_kernel(Pair const* pairs, std::size_t n, int num_gpus, Pair** partition_ptrs, std::size_t* write_offsets) {
  auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = tid; i < n; i += stride) {
    int pid = partition_id(pairs[i].first, num_gpus);
    auto ix = atomicAdd(reinterpret_cast<unsigned long long*>(&write_offsets[pid]), 1ULL);
    partition_ptrs[pid][ix] = pairs[i];
  }
}

template <typename Key>
__global__ void scatter_keys_kernel(Key const* keys, std::size_t n, int num_gpus, Key** key_ptrs, std::size_t** idx_ptrs, std::size_t* write_offsets) {
  auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = tid; i < n; i += stride) {
    int pid = partition_id(keys[i], num_gpus);
    auto ix = atomicAdd(reinterpret_cast<unsigned long long*>(&write_offsets[pid]), 1ULL);
    key_ptrs[pid][ix] = keys[i];
    idx_ptrs[pid][ix] = i;
  }
}

template <typename T>
__global__ void gather_results_kernel(T const* results, std::size_t const* index_map, std::size_t n, T* output) {
  auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = tid; i < n; i += stride) { output[index_map[i]] = results[i]; }
}

constexpr int kDefaultBlockSize = 256;
inline int grid_size_for(std::size_t n, int bs = kDefaultBlockSize) { return static_cast<int>((n + bs - 1) / bs); }

}}}
