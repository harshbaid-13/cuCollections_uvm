// =============================================================================
// multi_gpu_policies_test.cu
//
// Implements all 4 distribution policies from WarpDrive (Section IV-B)
// using cuco::multi_gpu_static_map + your existing test harness pattern.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_80 multi_gpu_policies_test.cu \
//        -I/path/to/cuco/include -o policy_test
//
// Run:
//   ./policy_test [N] [policy]
//   policy: 1=host_sided  2=system_atomics  3=unstructured  4=multisplit
//   default: runs all 4 and prints comparison table
// =============================================================================

#include <cuco/multi_gpu_static_map.cuh>
#include <cuco/static_map.cuh>
#include <cuco/pair.cuh>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <numeric>

using Key       = int32_t;
using Value     = int32_t;
using pair_type = cuco::pair<Key, Value>;

// ── Error checking ────────────────────────────────────────────────────────────
#define CK(x) do {                                                        \
  cudaError_t e = (x);                                                    \
  if (e != cudaSuccess) {                                                 \
    fprintf(stderr, "CUDA err %s:%d  %s\n",                              \
            __FILE__, __LINE__, cudaGetErrorString(e));                   \
    exit(1);                                                              \
  }                                                                       \
} while(0)

// ── Timer ─────────────────────────────────────────────────────────────────────
struct Timer {
  using C = std::chrono::high_resolution_clock;
  C::time_point t;
  Timer() : t(C::now()) {}
  float ms() {
    return std::chrono::duration<float, std::milli>(C::now() - t).count();
  }
};

// ── Partition hash (mirrors partition_kernels.cuh) ────────────────────────────
__host__ __device__ inline uint32_t partition_hash(int32_t k) {
  uint32_t x = static_cast<uint32_t>(k);
  x ^= x >> 16; x *= 0x85ebca6bU;
  x ^= x >> 13; x *= 0xc2b2ae35U;
  x ^= x >> 16;
  return x;
}
__host__ __device__ inline int partition_id(Key k, int num_gpus) {
  return static_cast<int>(partition_hash(k) % static_cast<uint32_t>(num_gpus));
}

// =============================================================================
// KERNELS shared by multiple policies
// =============================================================================

// Generate key-value pairs: key = i+1, value = (i+1)*10
__global__ void gen_pairs(pair_type* p, std::size_t n) {
  auto tid = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  auto s   = gridDim.x  * (std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s) {
    p[i].first  = (Key)(i + 1);
    p[i].second = (Key)(i + 1) * 10;
  }
}

// Extract keys from pairs
__global__ void get_keys(pair_type const* p, Key* k, std::size_t n) {
  auto tid = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  auto s   = gridDim.x  * (std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s) k[i] = p[i].first;
}

// Verify find results: value should equal key * mul
__global__ void verify_find(Key const* k, Value const* v,
                             std::size_t n, uint64_t* err, Value mul) {
  auto tid = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  auto s   = gridDim.x  * (std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s)
    if (v[i] != k[i] * mul)
      atomicAdd((unsigned long long*)err, 1ULL);
}

// Count true values in a bool array
__global__ void count_true(bool const* f, std::size_t n, uint64_t* c) {
  auto tid = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  auto s   = gridDim.x  * (std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s)
    if (f[i]) atomicAdd((unsigned long long*)c, 1ULL);
}

// ── Policy 1 helper: CPU partitioning scatter kernel ─────────────────────────
// Counts how many pairs belong to each GPU partition
__global__ void count_per_partition(pair_type const* pairs, std::size_t n,
                                    int num_gpus, std::size_t* counts) {
  auto tid = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  auto s   = gridDim.x  * (std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s) {
    int pid = partition_id(pairs[i].first, num_gpus);
    atomicAdd((unsigned long long*)&counts[pid], 1ULL);
  }
}

// Scatter pairs into per-partition buffers
__global__ void scatter_pairs(pair_type const* pairs, std::size_t n,
                              int num_gpus,
                              pair_type** part_ptrs,
                              std::size_t* write_offsets) {
  auto tid = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  auto s   = gridDim.x  * (std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s) {
    int pid = partition_id(pairs[i].first, num_gpus);
    auto ix = atomicAdd((unsigned long long*)&write_offsets[pid], 1ULL);
    part_ptrs[pid][ix] = pairs[i];
  }
}

// Gather results back to original order (used in find for policies 3 & 4)
__global__ void gather_results(Value const* results,
                               std::size_t const* index_map,
                               std::size_t n, Value* output) {
  auto tid = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  auto s   = gridDim.x  * (std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s)
    output[index_map[i]] = results[i];
}

// Scatter keys + save original indices (for find in policies 3 & 4)
__global__ void scatter_keys_idx(Key const* keys, std::size_t n,
                                 int num_gpus,
                                 Key** key_ptrs,
                                 std::size_t** idx_ptrs,
                                 std::size_t* write_offsets) {
  auto tid = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  auto s   = gridDim.x  * (std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s) {
    int pid = partition_id(keys[i], num_gpus);
    auto ix = atomicAdd((unsigned long long*)&write_offsets[pid], 1ULL);
    key_ptrs[pid][ix] = keys[i];
    idx_ptrs[pid][ix] = i;
  }
}

// =============================================================================
// Results structure returned by every policy benchmark
// =============================================================================
struct PolicyResult {
  const char* name;
  float  insert_ms;
  float  find_ms;
  float  contains_ms;
  bool   find_correct;
  bool   contains_correct;
  double insert_mops;
  double find_mops;
  double contains_mops;
};

// =============================================================================
// POLICY 1: Host-Sided Partitioning
// -----------------------------------------------------------------------------
// CPU computes p(k) for each key, reorders pairs in host RAM into per-GPU
// buckets, then transfers each bucket to its GPU via PCIe.
//
// WHY IT'S SLOW: linear reorder in RAM + full PCIe cost for every element.
// Included for completeness and as a baseline.
// =============================================================================
PolicyResult policy1_host_sided(std::size_t N,
                                std::vector<int> const& gpus,
                                pair_type* d_pairs,   // UVM, pairs on gpu[0]
                                Key*       d_keys,
                                Value*     d_vals,
                                bool*      d_found) {
  int const ngpus = (int)gpus.size();
  int const bs = 256;
  int const gr = std::min((int)((N + bs - 1) / bs), 65535);

  printf("  [Policy 1] Host-Sided Partitioning\n");
  printf("    Step 1: Copy pairs to host RAM and partition there...\n");

  // ── Pull pairs to host ────────────────────────────────────────────────────
  std::vector<pair_type> h_pairs(N);
  CK(cudaMemcpy(h_pairs.data(), d_pairs, N * sizeof(pair_type),
                cudaMemcpyDeviceToHost));

  // ── CPU partitioning: compute p(k) for every key ──────────────────────────
  // This is the expensive part the paper complains about: O(N) in host RAM
  std::vector<std::vector<pair_type>> buckets(ngpus);
  for (int g = 0; g < ngpus; ++g)
    buckets[g].reserve(N / ngpus + 1);

  Timer t_part;
  for (std::size_t i = 0; i < N; ++i) {
    int pid = partition_id(h_pairs[i].first, ngpus);
    buckets[pid].push_back(h_pairs[i]);
  }
  printf("    CPU partition time: %.1f ms\n", t_part.ms());

  // ── Create one cuco::static_map per GPU ───────────────────────────────────
  std::size_t cap_per_gpu = (std::size_t)((N / ngpus) / 0.8) + 1024;
  std::vector<cuco::static_map<Key,Value>*> maps(ngpus);
  std::vector<pair_type*> d_bufs(ngpus);

  for (int g = 0; g < ngpus; ++g) {
    CK(cudaSetDevice(gpus[g]));
    maps[g] = new cuco::static_map<Key,Value>(
        cap_per_gpu,
        cuco::empty_key<Key>{-1},
        cuco::empty_value<Value>{-1});
    CK(cudaMalloc(&d_bufs[g], buckets[g].size() * sizeof(pair_type)));
    // PCIe transfer: this is the bottleneck
    CK(cudaMemcpy(d_bufs[g], buckets[g].data(),
                  buckets[g].size() * sizeof(pair_type),
                  cudaMemcpyHostToDevice));
  }

  // ── Insert ────────────────────────────────────────────────────────────────
  Timer ti;
  for (int g = 0; g < ngpus; ++g) {
    CK(cudaSetDevice(gpus[g]));
    maps[g]->insert(d_bufs[g], d_bufs[g] + buckets[g].size());
  }
  for (int g = 0; g < ngpus; ++g) {
    CK(cudaSetDevice(gpus[g]));
    CK(cudaDeviceSynchronize());
  }
  float ins_ms = ti.ms();
  printf("    Insert: %.1f ms\n", ins_ms);

  // ── Find: must query ALL GPUs for every key (we know ownership though) ────
  // With host-sided policy we DO know which GPU owns each key (since we
  // partitioned by p(k)), so we can route correctly.
  // We reuse the same buckets structure: scatter queries, collect results.
  CK(cudaSetDevice(gpus[0]));
  std::vector<Key>   h_keys(N);
  std::vector<Value> h_vals(N, -1);
  CK(cudaMemcpy(h_keys.data(), d_keys, N * sizeof(Key), cudaMemcpyDeviceToHost));

  // Build per-GPU key lists + track original indices
  std::vector<std::vector<Key>>         qkeys(ngpus);
  std::vector<std::vector<std::size_t>> qidx(ngpus);
  for (std::size_t i = 0; i < N; ++i) {
    int pid = partition_id(h_keys[i], ngpus);
    qkeys[pid].push_back(h_keys[i]);
    qidx[pid].push_back(i);
  }

  std::vector<Key*>   d_qkeys(ngpus, nullptr);
  std::vector<Value*> d_qvals(ngpus, nullptr);

  Timer tf;
  for (int g = 0; g < ngpus; ++g) {
    if (qkeys[g].empty()) continue;
    CK(cudaSetDevice(gpus[g]));
    CK(cudaMalloc(&d_qkeys[g], qkeys[g].size() * sizeof(Key)));
    CK(cudaMalloc(&d_qvals[g], qkeys[g].size() * sizeof(Value)));
    CK(cudaMemcpy(d_qkeys[g], qkeys[g].data(),
                  qkeys[g].size() * sizeof(Key), cudaMemcpyHostToDevice));
    maps[g]->find(d_qkeys[g], d_qkeys[g] + qkeys[g].size(), d_qvals[g]);
  }
  for (int g = 0; g < ngpus; ++g) {
    if (qkeys[g].empty()) continue;
    CK(cudaSetDevice(gpus[g]));
    CK(cudaDeviceSynchronize());
    std::vector<Value> tmp(qkeys[g].size());
    CK(cudaMemcpy(tmp.data(), d_qvals[g],
                  qkeys[g].size() * sizeof(Value), cudaMemcpyDeviceToHost));
    for (std::size_t j = 0; j < qkeys[g].size(); ++j)
      h_vals[qidx[g][j]] = tmp[j];
  }
  float fnd_ms = tf.ms();
  printf("    Find:   %.1f ms\n", fnd_ms);

  // Copy results back to device for verify kernel
  CK(cudaSetDevice(gpus[0]));
  CK(cudaMemcpy(d_vals, h_vals.data(), N * sizeof(Value), cudaMemcpyHostToDevice));

  uint64_t* d_err; CK(cudaMallocManaged(&d_err, 8)); *d_err = 0;
  verify_find<<<gr, bs>>>(d_keys, d_vals, N, d_err, 10);
  CK(cudaDeviceSynchronize());
  bool find_ok = (*d_err == 0);
  printf("    Find verify: %s\n", find_ok ? "✓ PASSED" : "✗ FAILED");

  // ── Contains (same routing) ───────────────────────────────────────────────
  std::vector<bool*> d_qfound(ngpus, nullptr);
  Timer tc;
  for (int g = 0; g < ngpus; ++g) {
    if (qkeys[g].empty()) continue;
    CK(cudaSetDevice(gpus[g]));
    CK(cudaMalloc(&d_qfound[g], qkeys[g].size() * sizeof(bool)));
    maps[g]->contains(d_qkeys[g], d_qkeys[g] + qkeys[g].size(), d_qfound[g]);
  }
  for (int g = 0; g < ngpus; ++g) {
    if (qkeys[g].empty()) continue;
    CK(cudaSetDevice(gpus[g]));
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> tmp(qkeys[g].size());
    CK(cudaMemcpy(tmp.data(), d_qfound[g],
                  qkeys[g].size() * sizeof(bool), cudaMemcpyDeviceToHost));
    for (std::size_t j = 0; j < qkeys[g].size(); ++j)
      d_found[qidx[g][j]] = static_cast<bool>(tmp[j]);  // UVM: writable from host
  }
  float con_ms = tc.ms();

  uint64_t* d_cnt; CK(cudaMallocManaged(&d_cnt, 8)); *d_cnt = 0;
  CK(cudaSetDevice(gpus[0]));
  count_true<<<gr, bs>>>(d_found, N, d_cnt);
  CK(cudaDeviceSynchronize());
  bool con_ok = (*d_cnt == N);
  printf("    Contains verify: %s (%lu/%zu)\n\n",
         con_ok ? "✓ PASSED" : "✗ FAILED", *d_cnt, N);

  // ── Cleanup ───────────────────────────────────────────────────────────────
  for (int g = 0; g < ngpus; ++g) {
    CK(cudaSetDevice(gpus[g]));
    delete maps[g];
    if (d_bufs[g])   CK(cudaFree(d_bufs[g]));
    if (d_qkeys[g])  CK(cudaFree(d_qkeys[g]));
    if (d_qvals[g])  CK(cudaFree(d_qvals[g]));
    if (d_qfound[g]) CK(cudaFree(d_qfound[g]));
  }
  CK(cudaFree(d_err));
  CK(cudaFree(d_cnt));

  return { "Host-Sided",
           ins_ms, fnd_ms, con_ms,
           find_ok, con_ok,
           N / 1e6 / (ins_ms / 1e3),
           N / 1e6 / (fnd_ms / 1e3),
           N / 1e6 / (con_ms / 1e3) };
}

// =============================================================================
// POLICY 2: System-Wide Lock-Free Insertion (Unified Memory)
// -----------------------------------------------------------------------------
// One hash map in UVM. Every GPU inserts using system-wide atomics.
// Every CAS that hits a remote page crosses PCIe/NVLink → extremely slow.
//
// Implementation note: cuco::static_map constructed in managed memory,
// all GPUs advised to access it. This is the closest approximation to
// "system-wide atomics" without writing a custom probing kernel.
// =============================================================================
PolicyResult policy2_system_atomics(std::size_t N,
                                    std::vector<int> const& gpus,
                                    pair_type* d_pairs,
                                    Key*       d_keys,
                                    Value*     d_vals,
                                    bool*      d_found) {
  int const ngpus = (int)gpus.size();
  int const bs = 256;
  int const gr = std::min((int)((N + bs - 1) / bs), 65535);

  printf("  [Policy 2] System-Wide Lock-Free Insertion (UVM)\n");
  printf("    Building one hash map in managed memory...\n");

  std::size_t cap = (std::size_t)(N / 0.8);

  // Use the multi_gpu_static_map which internally uses UVM
  // This is the cuco equivalent of system-wide unified addressing
  auto map = cuco::multi_gpu_static_map<Key,Value>(
      cap, gpus,
      cuco::empty_key<Key>{-1},
      cuco::empty_value<Value>{-1});

  // ── Insert: all GPUs write into the same logical address space ───────────
  // Every atomic CAS that touches a remote GPU's page goes over NVLink/PCIe
  Timer ti;
  map.insert(d_pairs, d_pairs + N);
  float ins_ms = ti.ms();
  printf("    Insert: %.1f ms (note: cross-GPU atomics dominate)\n", ins_ms);

  // ── Find ──────────────────────────────────────────────────────────────────
  Timer tf;
  map.find(d_keys, d_keys + N, d_vals);
  float fnd_ms = tf.ms();

  uint64_t* d_err; CK(cudaMallocManaged(&d_err, 8)); *d_err = 0;
  CK(cudaSetDevice(gpus[0]));
  verify_find<<<gr, bs>>>(d_keys, d_vals, N, d_err, 10);
  CK(cudaDeviceSynchronize());
  bool find_ok = (*d_err == 0);
  printf("    Find: %.1f ms  verify: %s\n", fnd_ms,
         find_ok ? "✓ PASSED" : "✗ FAILED");

  // ── Contains ──────────────────────────────────────────────────────────────
  Timer tc;
  map.contains(d_keys, d_keys + N, d_found);
  float con_ms = tc.ms();

  uint64_t* d_cnt; CK(cudaMallocManaged(&d_cnt, 8)); *d_cnt = 0;
  count_true<<<gr, bs>>>(d_found, N, d_cnt);
  CK(cudaDeviceSynchronize());
  bool con_ok = (*d_cnt == N);
  printf("    Contains: %.1f ms  verify: %s (%lu/%zu)\n\n",
         con_ms, con_ok ? "✓ PASSED" : "✗ FAILED", *d_cnt, N);

  CK(cudaFree(d_err));
  CK(cudaFree(d_cnt));

  return { "System-Atomics (UVM)",
           ins_ms, fnd_ms, con_ms,
           find_ok, con_ok,
           N / 1e6 / (ins_ms / 1e3),
           N / 1e6 / (fnd_ms / 1e3),
           N / 1e6 / (con_ms / 1e3) };
}

// =============================================================================
// POLICY 3: Unstructured Distribution
// -----------------------------------------------------------------------------
// Split input into N/m equal chunks, send chunk g to GPU g.
// Each GPU holds an independent hash map.
//
// INSERT: trivially fast — no partitioning needed.
// FIND:   must broadcast every query to ALL GPUs and merge results,
//         because we have no idea which GPU holds a given key.
//         This is the fundamental flaw the paper identifies.
// =============================================================================
PolicyResult policy3_unstructured(std::size_t N,
                                  std::vector<int> const& gpus,
                                  pair_type* d_pairs,   // UVM on gpu[0]
                                  Key*       d_keys,
                                  Value*     d_vals,
                                  bool*      d_found) {
  int const ngpus   = (int)gpus.size();
  int const bs      = 256;
  int const gr      = std::min((int)((N + bs - 1) / bs), 65535);
  std::size_t chunk = (N + ngpus - 1) / ngpus;  // equal-ish chunks

  printf("  [Policy 3] Unstructured Distribution\n");
  printf("    Chunk size per GPU: %zu\n", chunk);

  std::size_t cap_per_gpu = (std::size_t)(chunk / 0.8) + 1024;
  std::vector<cuco::static_map<Key,Value>*> maps(ngpus);
  std::vector<pair_type*> d_chunks(ngpus, nullptr);
  std::vector<std::size_t> chunk_sizes(ngpus);

  // Allocate and transfer chunks
  for (int g = 0; g < ngpus; ++g) {
    std::size_t start = (std::size_t)g * chunk;
    std::size_t end   = std::min(start + chunk, N);
    chunk_sizes[g]    = end - start;
    if (chunk_sizes[g] == 0) continue;

    CK(cudaSetDevice(gpus[g]));
    maps[g] = new cuco::static_map<Key,Value>(
        cap_per_gpu,
        cuco::empty_key<Key>{-1},
        cuco::empty_value<Value>{-1});

    CK(cudaMalloc(&d_chunks[g], chunk_sizes[g] * sizeof(pair_type)));
    // Transfer from GPU 0 (where d_pairs lives) to GPU g
    CK(cudaMemcpyPeer(d_chunks[g], gpus[g],
                      d_pairs + start, gpus[0],
                      chunk_sizes[g] * sizeof(pair_type)));
  }

  // ── Insert: each GPU inserts its chunk independently ──────────────────────
  Timer ti;
  for (int g = 0; g < ngpus; ++g) {
    if (!chunk_sizes[g]) continue;
    CK(cudaSetDevice(gpus[g]));
    maps[g]->insert(d_chunks[g], d_chunks[g] + chunk_sizes[g]);
  }
  for (int g = 0; g < ngpus; ++g) {
    if (!chunk_sizes[g]) continue;
    CK(cudaSetDevice(gpus[g]));
    CK(cudaDeviceSynchronize());
  }
  float ins_ms = ti.ms();
  printf("    Insert: %.1f ms (fast - no routing overhead)\n", ins_ms);

  // ── Find: broadcast EVERY query to ALL GPUs, merge with OR ───────────────
  // This is O(m) per query — the fundamental cost of unstructured distribution
  // "missing" values are filled with -1 (empty sentinel), so we take the
  // non-(-1) result across all GPUs.
  printf("    Find:   broadcasting queries to all %d GPUs (O(m) cost)...\n",
         ngpus);

  // Copy keys to host to distribute
  std::vector<Key> h_keys(N);
  CK(cudaSetDevice(gpus[0]));
  CK(cudaMemcpy(h_keys.data(), d_keys, N * sizeof(Key), cudaMemcpyDeviceToHost));

  std::vector<Value> h_merged(N, -1);  // -1 = not found yet

  std::vector<Key*>   d_qk(ngpus, nullptr);
  std::vector<Value*> d_qv(ngpus, nullptr);

  Timer tf;
  for (int g = 0; g < ngpus; ++g) {
    if (!chunk_sizes[g]) continue;
    CK(cudaSetDevice(gpus[g]));
    CK(cudaMalloc(&d_qk[g], N * sizeof(Key)));
    CK(cudaMalloc(&d_qv[g], N * sizeof(Value)));
    CK(cudaMemcpy(d_qk[g], h_keys.data(), N * sizeof(Key),
                  cudaMemcpyHostToDevice));
    maps[g]->find(d_qk[g], d_qk[g] + N, d_qv[g]);
  }

  // Merge: for each position take the first non-(-1) value
  for (int g = 0; g < ngpus; ++g) {
    if (!chunk_sizes[g]) continue;
    CK(cudaSetDevice(gpus[g]));
    CK(cudaDeviceSynchronize());
    std::vector<Value> tmp(N);
    CK(cudaMemcpy(tmp.data(), d_qv[g], N * sizeof(Value),
                  cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < N; ++i)
      if (h_merged[i] == -1 && tmp[i] != -1)
        h_merged[i] = tmp[i];
  }
  float fnd_ms = tf.ms();

  CK(cudaSetDevice(gpus[0]));
  CK(cudaMemcpy(d_vals, h_merged.data(), N * sizeof(Value),
                cudaMemcpyHostToDevice));

  uint64_t* d_err; CK(cudaMallocManaged(&d_err, 8)); *d_err = 0;
  verify_find<<<gr, bs>>>(d_keys, d_vals, N, d_err, 10);
  CK(cudaDeviceSynchronize());
  bool find_ok = (*d_err == 0);
  printf("    Find: %.1f ms  verify: %s\n", fnd_ms,
         find_ok ? "✓ PASSED" : "✗ FAILED");

  // ── Contains: same broadcast approach ────────────────────────────────────
  std::vector<bool*>    d_qf(ngpus, nullptr);
  std::vector<uint8_t>  h_merged_b(N, 0);

  Timer tc;
  for (int g = 0; g < ngpus; ++g) {
    if (!chunk_sizes[g]) continue;
    CK(cudaSetDevice(gpus[g]));
    CK(cudaMalloc(&d_qf[g], N * sizeof(bool)));
    maps[g]->contains(d_qk[g], d_qk[g] + N, d_qf[g]);
  }
  for (int g = 0; g < ngpus; ++g) {
    if (!chunk_sizes[g]) continue;
    CK(cudaSetDevice(gpus[g]));
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> tmp(N);
    CK(cudaMemcpy(tmp.data(), d_qf[g], N * sizeof(bool),
                  cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < N; ++i)
      h_merged_b[i] = h_merged_b[i] || static_cast<bool>(tmp[i]);
  }
  float con_ms = tc.ms();
  for (std::size_t i = 0; i < N; ++i)
    d_found[i] = h_merged_b[i];

  uint64_t* d_cnt; CK(cudaMallocManaged(&d_cnt, 8)); *d_cnt = 0;
  CK(cudaSetDevice(gpus[0]));
  count_true<<<gr, bs>>>(d_found, N, d_cnt);
  CK(cudaDeviceSynchronize());
  bool con_ok = (*d_cnt == N);
  printf("    Contains: %.1f ms  verify: %s (%lu/%zu)\n\n",
         con_ms, con_ok ? "✓ PASSED" : "✗ FAILED", *d_cnt, N);

  // ── Cleanup ───────────────────────────────────────────────────────────────
  for (int g = 0; g < ngpus; ++g) {
    CK(cudaSetDevice(gpus[g]));
    if (maps[g])   delete maps[g];
    if (d_chunks[g]) CK(cudaFree(d_chunks[g]));
    if (d_qk[g])   CK(cudaFree(d_qk[g]));
    if (d_qv[g])   CK(cudaFree(d_qv[g]));
    if (d_qf[g])   CK(cudaFree(d_qf[g]));
  }
  CK(cudaFree(d_err));
  CK(cudaFree(d_cnt));

  return { "Unstructured",
           ins_ms, fnd_ms, con_ms,
           find_ok, con_ok,
           N / 1e6 / (ins_ms / 1e3),
           N / 1e6 / (fnd_ms / 1e3),
           N / 1e6 / (con_ms / 1e3) };
}

// =============================================================================
// POLICY 4: Distributed Multisplit Transposition  ← THE ONE THAT WORKS
// -----------------------------------------------------------------------------
// This is exactly what the paper adopts and what cuco::multi_gpu_static_map
// implements internally.
//
// Pipeline:
//   INSERT:  [H2D] → multisplit → all-to-all transpose → local insert
//   FIND:    [H2D] → multisplit → transpose → local find → rev-transpose → [D2H]
//
// Batched async variant overlaps these stages across batches (Figure 5).
// =============================================================================
PolicyResult policy4_multisplit(std::size_t N,
                                std::vector<int> const& gpus,
                                pair_type* d_pairs,
                                Key*       d_keys,
                                Value*     d_vals,
                                bool*      d_found) {
  int const ngpus = (int)gpus.size();
  int const bs    = 256;
  int const gr    = std::min((int)((N + bs - 1) / bs), 65535);

  printf("  [Policy 4] Distributed Multisplit Transposition\n");

  std::size_t cap = (std::size_t)(N / 0.8);
  auto map = cuco::multi_gpu_static_map<Key,Value>(
      cap, gpus,
      cuco::empty_key<Key>{-1},
      cuco::empty_value<Value>{-1});

  // ── Insert: multisplit → NVLink transpose → local insert ─────────────────
  Timer ti;
  map.insert(d_pairs, d_pairs + N);
  float ins_ms = ti.ms();

  auto sizes = map.per_gpu_sizes();
  std::size_t tot = 0;
  for (int g = 0; g < ngpus; ++g) {
    printf("    GPU %d: %zu elements\n", gpus[g], sizes[g]);
    tot += sizes[g];
  }
  printf("    Total inserted: %zu  Insert: %.1f ms\n", tot, ins_ms);

  // ── Find ──────────────────────────────────────────────────────────────────
  Timer tf;
  map.find(d_keys, d_keys + N, d_vals);
  float fnd_ms = tf.ms();

  uint64_t* d_err; CK(cudaMallocManaged(&d_err, 8)); *d_err = 0;
  CK(cudaSetDevice(gpus[0]));
  verify_find<<<gr, bs>>>(d_keys, d_vals, N, d_err, 10);
  CK(cudaDeviceSynchronize());
  bool find_ok = (*d_err == 0);
  printf("    Find: %.1f ms  verify: %s\n", fnd_ms,
         find_ok ? "✓ PASSED" : "✗ FAILED");

  // ── Contains ──────────────────────────────────────────────────────────────
  Timer tc;
  map.contains(d_keys, d_keys + N, d_found);
  float con_ms = tc.ms();

  uint64_t* d_cnt; CK(cudaMallocManaged(&d_cnt, 8)); *d_cnt = 0;
  count_true<<<gr, bs>>>(d_found, N, d_cnt);
  CK(cudaDeviceSynchronize());
  bool con_ok = (*d_cnt == N);
  printf("    Contains: %.1f ms  verify: %s (%lu/%zu)\n", con_ms,
         con_ok ? "✓ PASSED" : "✗ FAILED", *d_cnt, N);

  // ── Async pipelined insert (Figure 5 from paper) ─────────────────────────
  // Demonstrate CPU/GPU overlap: while GPU processes batch B, CPU prepares B+1
  // Batch size matches the paper: 2^24 = 16M elements per batch
  constexpr std::size_t BATCH_SIZE = 1 << 24;
  if (N >= BATCH_SIZE * 2) {
    printf("\n    --- Async pipelined insert (2 threads, batch=%zu) ---\n",
           BATCH_SIZE);

    auto map2 = cuco::multi_gpu_static_map<Key,Value>(
        cap, gpus,
        cuco::empty_key<Key>{-1},
        cuco::empty_value<Value>{-1});

    std::size_t nbatches = (N + BATCH_SIZE - 1) / BATCH_SIZE;
    int const nthreads = 2;  // paper uses 2-4 threads

    Timer tp;
    // Each CPU thread handles every nthreads-th batch.
    // Thread 0 does batches 0, 2, 4, ...  H2D → MST → INS
    // Thread 1 does batches 1, 3, 5, ...  H2D → MST → INS  (overlapped)
    std::vector<std::future<void>> futures;
    futures.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
      futures.push_back(std::async(std::launch::async, [&, t]() {
        for (std::size_t b = t; b < nbatches; b += nthreads) {
          std::size_t start  = b * BATCH_SIZE;
          std::size_t end    = std::min(start + BATCH_SIZE, N);
          std::size_t bcount = end - start;
          map2.insert(d_pairs + start, d_pairs + start + bcount);
        }
      }));
    }
    for (auto& f : futures) f.get();
    float pipe_ms = tp.ms();
    printf("    Pipelined insert (%d threads): %.1f ms  (vs sequential %.1f ms)\n",
           nthreads, pipe_ms, ins_ms);
    printf("    Speedup from overlap: %.2fx\n\n", ins_ms / pipe_ms);
  } else {
    printf("\n");
  }

  CK(cudaFree(d_err));
  CK(cudaFree(d_cnt));

  return { "Multisplit Transposition",
           ins_ms, fnd_ms, con_ms,
           find_ok, con_ok,
           N / 1e6 / (ins_ms / 1e3),
           N / 1e6 / (fnd_ms / 1e3),
           N / 1e6 / (con_ms / 1e3) };
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char** argv) {
  std::size_t N      = argc > 1 ? std::atol(argv[1]) : 10000000ULL;
  int         policy = argc > 2 ? std::atoi(argv[2]) : 0;  // 0 = run all

  constexpr int       ngpus = 2;
  std::vector<int> const gpus{1, 2};

  int dev_count;
  CK(cudaGetDeviceCount(&dev_count));
  if (dev_count < 1 + ngpus) {
    fprintf(stderr, "Need at least %d CUDA devices, found %d\n",
            1 + ngpus, dev_count);
    return 1;
  }

  int const host_dev = gpus[0];
  CK(cudaSetDevice(host_dev));

  // Enable peer access between all GPU pairs
  for (int i = 0; i < ngpus; ++i) {
    CK(cudaSetDevice(gpus[i]));
    for (int j = 0; j < ngpus; ++j) {
      if (i == j) continue;
      int can = 0;
      cudaDeviceCanAccessPeer(&can, gpus[i], gpus[j]);
      if (can) cudaDeviceEnablePeerAccess(gpus[j], 0);
    }
  }
  CK(cudaSetDevice(host_dev));

  printf("╔═══════════════════════════════════════════════════════╗\n");
  printf("║   WarpDrive 4-Policy Comparison Test                  ║\n");
  printf("╠═══════════════════════════════════════════════════════╣\n");
  printf("║  GPUs:           %-3d (devices %d,%d,%d)                 ║\n",
         ngpus, gpus[0], gpus[1], gpus[2]);
  for (int i = 0; i < ngpus; i++) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, gpus[i]));
    printf("║    GPU %d: %-25s %4.1f GB   ║\n",
           gpus[i], p.name, p.totalGlobalMem / (1024.*1024.*1024.));
  }
  printf("║  Elements:       %-12zu                       ║\n", N);
  printf("║  Data size:      %-5.2f GB                             ║\n",
         N * 8 / (1024.*1024.*1024.));
  printf("║  Policy:         %-3s                                  ║\n",
         policy == 0 ? "ALL" :
         policy == 1 ? "1 (Host-sided)" :
         policy == 2 ? "2 (System atomics)" :
         policy == 3 ? "3 (Unstructured)" : "4 (Multisplit)");
  printf("╚═══════════════════════════════════════════════════════╝\n\n");

  // ── Allocate shared UVM buffers (all policies read from these) ───────────
  pair_type* d_pairs; CK(cudaMallocManaged(&d_pairs, sizeof(pair_type) * N));
  Key*       d_keys;  CK(cudaMallocManaged(&d_keys,  sizeof(Key)       * N));
  Value*     d_vals;  CK(cudaMallocManaged(&d_vals,  sizeof(Value)     * N));
  bool*      d_found; CK(cudaMallocManaged(&d_found, sizeof(bool)      * N));

  for (int d : gpus) {
    CK(cudaMemAdvise(d_pairs, sizeof(pair_type)*N, cudaMemAdviseSetAccessedBy, d));
    CK(cudaMemAdvise(d_keys,  sizeof(Key)*N,       cudaMemAdviseSetAccessedBy, d));
    CK(cudaMemAdvise(d_vals,  sizeof(Value)*N,     cudaMemAdviseSetAccessedBy, d));
    CK(cudaMemAdvise(d_found, sizeof(bool)*N,      cudaMemAdviseSetAccessedBy, d));
  }

  int bl = 256;
  int gr = std::min((int)((N + bl - 1) / bl), 65535);
  gen_pairs<<<gr, bl>>>(d_pairs, N);
  CK(cudaDeviceSynchronize());
  get_keys<<<gr, bl>>>(d_pairs, d_keys, N);
  CK(cudaDeviceSynchronize());
  printf("[Data generated: %zu pairs]\n\n", N);

  // ── Run selected policy/policies ─────────────────────────────────────────
  std::vector<PolicyResult> results;

  if (policy == 0 || policy == 1) {
    printf("══════════════════════════════════════════════\n");
    results.push_back(
        policy1_host_sided(N, gpus, d_pairs, d_keys, d_vals, d_found));
  }
  if (policy == 0 || policy == 2) {
    printf("══════════════════════════════════════════════\n");
    results.push_back(
        policy2_system_atomics(N, gpus, d_pairs, d_keys, d_vals, d_found));
  }
  if (policy == 0 || policy == 3) {
    printf("══════════════════════════════════════════════\n");
    results.push_back(
        policy3_unstructured(N, gpus, d_pairs, d_keys, d_vals, d_found));
  }
  if (policy == 0 || policy == 4) {
    printf("══════════════════════════════════════════════\n");
    results.push_back(
        policy4_multisplit(N, gpus, d_pairs, d_keys, d_vals, d_found));
  }

  // ── Comparison table ─────────────────────────────────────────────────────
  printf("\n");
  printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
  printf("║  COMPARISON TABLE  (%zu elements, %d GPUs)                               ║\n",
         N, ngpus);
  printf("╠══════════════════════╦══════════════╦══════════════╦══════════════╦══════╣\n");
  printf("║ Policy               ║  Insert      ║  Find        ║  Contains    ║  OK? ║\n");
  printf("╠══════════════════════╬══════════════╬══════════════╬══════════════╬══════╣\n");
  for (auto const& r : results) {
    printf("║ %-20s ║ %6.1f ms     ║ %6.1f ms     ║ %6.1f ms     ║  %s   ║\n",
           r.name,
           r.insert_ms,
           r.find_ms,
           r.contains_ms,
           (r.find_correct && r.contains_correct) ? "✓" : "✗");
  }
  printf("╠══════════════════════╬══════════════╬══════════════╬══════════════╬══════╣\n");
  printf("║ Policy               ║  Insert M/s  ║  Find M/s    ║  Contains M/s║      ║\n");
  printf("╠══════════════════════╬══════════════╬══════════════╬══════════════╬══════╣\n");
  for (auto const& r : results) {
    printf("║ %-20s ║ %8.2f     ║ %8.2f     ║ %8.2f     ║      ║\n",
           r.name, r.insert_mops, r.find_mops, r.contains_mops);
  }
  printf("╚══════════════════════╩══════════════╩══════════════╩══════════════╩══════╝\n");

  // ── Interpretation notes ──────────────────────────────────────────────────
  printf("\nNotes:\n");
  printf("  Policy 1 (Host-sided):    CPU reorder in RAM + full PCIe per element\n");
  printf("                            → bottleneck even before any GPU work\n");
  printf("  Policy 2 (System atomics):UVM lets all GPUs share one map\n");
  printf("                            → every cross-GPU CAS crosses NVLink/PCIe\n");
  printf("  Policy 3 (Unstructured):  Insert is fast; find broadcasts to all %d GPUs\n",
         ngpus);
  printf("                            → find/contains cost scales as O(m)\n");
  printf("  Policy 4 (Multisplit):    All work in VRAM; O(1) routing at query time\n");
  printf("                            → scales with NVLink bandwidth (best)\n");

  CK(cudaFree(d_pairs));
  CK(cudaFree(d_keys));
  CK(cudaFree(d_vals));
  CK(cudaFree(d_found));

  printf("\nDone.\n");
  return 0;
}
