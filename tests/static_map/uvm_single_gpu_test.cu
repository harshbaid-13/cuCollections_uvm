/**
 * @file uvm_single_gpu_test.cu
 * @brief Standalone test for cuCollections with UVM (cudaMallocManaged) allocator.
 *
 * Build (from tests/static_map/):
 *   nvcc -std=c++17 --expt-extended-lambda -arch=sm_75 \
 *        -I../. -I../../include \
 *        -I/usr/local/cuda-13.2/targets/x86_64-linux/include/cccl \
 *        uvm_single_gpu_test.cu -o uvm_test
 *
 * Run:
 *   ./uvm_test                        # default 50M elements
 *   ./uvm_test 200000000              # 200M elements
 *   ./uvm_test 200000000 10000000     # 200M elements, 10M batch size
 */

#include <cuco/static_map.cuh>
#include <cuco/pair.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/allocator.hpp>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ─── Types ───────────────────────────────────────────────────────────────────
using Key       = int32_t;
using Value     = int32_t;
using pair_type = cuco::pair<Key, Value>;
using size_type = std::size_t;

// ─── CUDA 13+ compatible UVM helpers ─────────────────────────────────────────
// CUDA 13 changed cudaMemAdvise/cudaMemPrefetchAsync to use cudaMemLocation
// instead of a plain int device ID.

inline void uvm_set_accessed_by(void* ptr, size_t bytes, int device_id)
{
  cudaMemLocation loc = {};
  loc.type = cudaMemLocationTypeDevice;
  loc.id   = device_id;
  cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, loc);
}

inline void uvm_prefetch(void* ptr, size_t bytes, int device_id, cudaStream_t stream = 0)
{
  cudaMemLocation loc = {};
  loc.type = cudaMemLocationTypeDevice;
  loc.id   = device_id;
  cudaMemPrefetchAsync(ptr, bytes, loc, 0, stream);
}

// ─── Timer ───────────────────────────────────────────────────────────────────
struct Timer {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point t0;
  Timer() : t0(clock::now()) {}
  float ms() const {
    return std::chrono::duration<float, std::milli>(clock::now() - t0).count();
  }
};

// ─── CUDA error check ────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// ─── GPU kernels ─────────────────────────────────────────────────────────────

__global__ void generate_pairs_kernel(pair_type* pairs, size_type n)
{
  size_type tid    = blockIdx.x * (size_type)blockDim.x + threadIdx.x;
  size_type stride = gridDim.x * (size_type)blockDim.x;
  for (size_type i = tid; i < n; i += stride) {
    Key k           = static_cast<Key>(i + 1);
    pairs[i].first  = k;
    pairs[i].second = k * 10;
  }
}

__global__ void extract_keys_kernel(pair_type const* pairs, Key* keys, size_type n)
{
  size_type tid    = blockIdx.x * (size_type)blockDim.x + threadIdx.x;
  size_type stride = gridDim.x * (size_type)blockDim.x;
  for (size_type i = tid; i < n; i += stride) {
    keys[i] = pairs[i].first;
  }
}

__global__ void verify_find_kernel(Key const* keys, Value const* values,
                                    size_type n, uint64_t* errors,
                                    Value expected_multiplier)
{
  size_type tid    = blockIdx.x * (size_type)blockDim.x + threadIdx.x;
  size_type stride = gridDim.x * (size_type)blockDim.x;
  for (size_type i = tid; i < n; i += stride) {
    Value expected = keys[i] * expected_multiplier;
    if (values[i] != expected) {
      atomicAdd((unsigned long long*)errors, 1ULL);
    }
  }
}

__global__ void count_true_kernel(bool const* flags, size_type n, uint64_t* count)
{
  size_type tid    = blockIdx.x * (size_type)blockDim.x + threadIdx.x;
  size_type stride = gridDim.x * (size_type)blockDim.x;
  for (size_type i = tid; i < n; i += stride) {
    if (flags[i]) atomicAdd((unsigned long long*)count, 1ULL);
  }
}

__global__ void update_values_kernel(pair_type* pairs, size_type n, Value new_multiplier)
{
  size_type tid    = blockIdx.x * (size_type)blockDim.x + threadIdx.x;
  size_type stride = gridDim.x * (size_type)blockDim.x;
  for (size_type i = tid; i < n; i += stride) {
    pairs[i].second = pairs[i].first * new_multiplier;
  }
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
  size_type num_elements = (argc > 1) ? std::atol(argv[1]) : 50000000ULL;
  uint32_t  batch_size   = (argc > 2) ? std::atoi(argv[2]) : 10000000;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  double data_bytes = num_elements * sizeof(pair_type);
  double table_est  = (num_elements / 0.9) * sizeof(pair_type) / 0.8;

  printf("╔═══════════════════════════════════════════════════════╗\n");
  printf("║   cuCollections UVM Single-GPU Test                  ║\n");
  printf("╠═══════════════════════════════════════════════════════╣\n");
  printf("║  GPU:            %-36s ║\n", prop.name);
  printf("║  VRAM:           %-5.1f GB                              ║\n",
         prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("║  Elements:       %-12zu                       ║\n", num_elements);
  printf("║  Data size:      %-5.2f GB                             ║\n",
         data_bytes / (1024.0 * 1024.0 * 1024.0));
  printf("║  Est table mem:  %-5.2f GB                             ║\n",
         table_est / (1024.0 * 1024.0 * 1024.0));
  printf("║  Batch size:     %-12u                       ║\n", batch_size);
  if (table_est > prop.totalGlobalMem) {
    printf("║  ⚠  OVERSUBSCRIPTION MODE — table > VRAM             ║\n");
  }
  printf("╚═══════════════════════════════════════════════════════╝\n\n");

  // ──────────────────────────────────────────────────────────
  //  1. Create the map (UVM allocator kicks in here)
  // ──────────────────────────────────────────────────────────
  printf("[1/6] Creating static_map with UVM allocator...\n");
  Timer t_create;

  using probe = cuco::double_hashing<16, cuco::xxhash_32<Key>, cuco::xxhash_32<Key>>;
  auto map = cuco::static_map<Key, Value,
                               cuco::extent<size_type>,
                               cuda::thread_scope_device,
                               cuda::std::equal_to<Key>,
                               probe,
                               cuco::cuda_allocator<cuda::std::byte>,
                               cuco::storage<2>>{
    num_elements, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};

  CUDA_CHECK(cudaDeviceSynchronize());
  printf("       Done in %.1f ms\n\n", t_create.ms());

  // ──────────────────────────────────────────────────────────
  //  2. Generate test data (UVM)
  // ──────────────────────────────────────────────────────────
  printf("[2/6] Generating %zu test pairs (UVM allocated)...\n", num_elements);
  pair_type* d_pairs = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_pairs, sizeof(pair_type) * num_elements));
  uvm_set_accessed_by(d_pairs, sizeof(pair_type) * num_elements, 0);
  uvm_prefetch(d_pairs, sizeof(pair_type) * num_elements, 0);

  int block = 256;
  int grid  = std::min((int)((num_elements + block - 1) / block), 65535);
  generate_pairs_kernel<<<grid, block>>>(d_pairs, num_elements);
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("       Done.\n\n");

  // ──────────────────────────────────────────────────────────
  //  3. INSERT (batched)
  // ──────────────────────────────────────────────────────────
  printf("[3/6] Inserting %zu pairs (batch_size=%u)...\n", num_elements, batch_size);
  Timer t_insert;

  size_type inserted_total = 0;
  uint32_t batch_num       = 0;
  while (inserted_total < num_elements) {
    size_type this_batch = std::min((size_type)batch_size, num_elements - inserted_total);
    pair_type* batch_ptr = d_pairs + inserted_total;

    uvm_set_accessed_by(batch_ptr, this_batch * sizeof(pair_type), 0);
    uvm_prefetch(batch_ptr, this_batch * sizeof(pair_type), 0);

    map.insert_or_assign(batch_ptr, batch_ptr + this_batch);

    inserted_total += this_batch;
    batch_num++;
    if (batch_num % 5 == 0 || inserted_total == num_elements) {
      printf("       Batch %u done — %zu / %zu (%.0f%%)\n",
             batch_num, inserted_total, num_elements,
             100.0 * inserted_total / num_elements);
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  float insert_ms = t_insert.ms();
  printf("       Insert total: %.1f ms (%.2f M ops/sec)\n\n",
         insert_ms, (num_elements / 1e6) / (insert_ms / 1e3));

  // ──────────────────────────────────────────────────────────
  //  4. FIND + verify
  // ──────────────────────────────────────────────────────────
  printf("[4/6] Finding all %zu keys...\n", num_elements);

  Key* d_keys = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_keys, sizeof(Key) * num_elements));
  extract_keys_kernel<<<grid, block>>>(d_pairs, d_keys, num_elements);
  CUDA_CHECK(cudaDeviceSynchronize());

  Value* d_values_out = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_values_out, sizeof(Value) * num_elements));
  uvm_set_accessed_by(d_keys, sizeof(Key) * num_elements, 0);
  uvm_set_accessed_by(d_values_out, sizeof(Value) * num_elements, 0);

  Timer t_find;
  map.find(d_keys, d_keys + num_elements, d_values_out);
  CUDA_CHECK(cudaDeviceSynchronize());
  float find_ms = t_find.ms();
  printf("       Find: %.1f ms (%.2f M ops/sec)\n", find_ms,
         (num_elements / 1e6) / (find_ms / 1e3));

  uint64_t* d_errors = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_errors, sizeof(uint64_t)));
  *d_errors = 0;
  verify_find_kernel<<<grid, block>>>(d_keys, d_values_out, num_elements, d_errors, 10);
  CUDA_CHECK(cudaDeviceSynchronize());

  if (*d_errors == 0) {
    printf("       ✓ Find verification PASSED\n\n");
  } else {
    printf("       ✗ Find verification FAILED — %lu errors\n\n", *d_errors);
  }

  // ──────────────────────────────────────────────────────────
  //  5. CONTAINS
  // ──────────────────────────────────────────────────────────
  printf("[5/6] Contains check for %zu keys...\n", num_elements);

  bool* d_found = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_found, sizeof(bool) * num_elements));
  uvm_set_accessed_by(d_found, sizeof(bool) * num_elements, 0);

  Timer t_contains;
  map.contains(d_keys, d_keys + num_elements, d_found);
  CUDA_CHECK(cudaDeviceSynchronize());
  float contains_ms = t_contains.ms();

  uint64_t* d_count = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_count, sizeof(uint64_t)));
  *d_count = 0;
  count_true_kernel<<<grid, block>>>(d_found, num_elements, d_count);
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("       Contains: %.1f ms (%.2f M ops/sec)\n", contains_ms,
         (num_elements / 1e6) / (contains_ms / 1e3));
  if (*d_count == num_elements) {
    printf("       ✓ All %zu keys found\n\n", num_elements);
  } else {
    printf("       ✗ Only %lu / %zu keys found\n\n", *d_count, num_elements);
  }

  // ──────────────────────────────────────────────────────────
  //  6. INSERT_OR_ASSIGN (update) + verify
  // ──────────────────────────────────────────────────────────
  printf("[6/6] Insert-or-assign (update values to key*20)...\n");

  update_values_kernel<<<grid, block>>>(d_pairs, num_elements, 20);
  CUDA_CHECK(cudaDeviceSynchronize());

  Timer t_upsert;
  inserted_total = 0;
  batch_num      = 0;
  while (inserted_total < num_elements) {
    size_type this_batch = std::min((size_type)batch_size, num_elements - inserted_total);
    pair_type* batch_ptr = d_pairs + inserted_total;
    map.insert_or_assign(batch_ptr, batch_ptr + this_batch);
    inserted_total += this_batch;
    batch_num++;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  float upsert_ms = t_upsert.ms();
  printf("       Update: %.1f ms (%.2f M ops/sec)\n", upsert_ms,
         (num_elements / 1e6) / (upsert_ms / 1e3));

  map.find(d_keys, d_keys + num_elements, d_values_out);
  CUDA_CHECK(cudaDeviceSynchronize());

  *d_errors = 0;
  verify_find_kernel<<<grid, block>>>(d_keys, d_values_out, num_elements, d_errors, 20);
  CUDA_CHECK(cudaDeviceSynchronize());

  if (*d_errors == 0) {
    printf("       ✓ Updated values verification PASSED\n\n");
  } else {
    printf("       ✗ Updated values verification FAILED — %lu errors\n\n", *d_errors);
  }

  // ──────────────────────────────────────────────────────────
  //  Summary
  // ──────────────────────────────────────────────────────────
  printf("═══════════════════════════════════════════════════════\n");
  printf("  RESULTS SUMMARY\n");
  printf("═══════════════════════════════════════════════════════\n");
  printf("  Insert:           %8.1f ms  (%6.2f M ops/sec)\n",
         insert_ms, (num_elements / 1e6) / (insert_ms / 1e3));
  printf("  Find:             %8.1f ms  (%6.2f M ops/sec)\n",
         find_ms, (num_elements / 1e6) / (find_ms / 1e3));
  printf("  Contains:         %8.1f ms  (%6.2f M ops/sec)\n",
         contains_ms, (num_elements / 1e6) / (contains_ms / 1e3));
  printf("  Insert-or-assign: %8.1f ms  (%6.2f M ops/sec)\n",
         upsert_ms, (num_elements / 1e6) / (upsert_ms / 1e3));
  printf("═══════════════════════════════════════════════════════\n");

  CUDA_CHECK(cudaFree(d_pairs));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values_out));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_errors));
  CUDA_CHECK(cudaFree(d_count));

  printf("\nAll tests complete.\n");
  return 0;
}
