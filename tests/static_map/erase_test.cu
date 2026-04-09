/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <test_utils.hpp>

#include <cuco/static_map.cuh>

#include <cuda/functional>
#include <cuda/std/tuple>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_session.hpp>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

using std::string;
using size_type       = std::size_t;
using HRClock         = std::chrono::high_resolution_clock;
using DurationFloatMS = std::chrono::duration<float, std::milli>;


// -----------------------------------------------------------------------------
// Global CLI-configurable parameters
// -----------------------------------------------------------------------------
static string g_insert_path =
  "/data/heterods-trace/insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE_5e8_Random_Clusters.bin";

static string g_delete_path =
  "/data/heterods-trace/insert_trace-400e7-100-add-no-dup-DENSE_UNIQUE.bin";

static size_type g_num_keys = 2000000000;


// -----------------------------------------------------------------------------
// Benchmark function
// -----------------------------------------------------------------------------
template <typename Map>
float test_unique_sequence(
  Map& map,
  size_type num_keys,
  size_type num_ins,
  const std::string& insert_path,
  const std::string& delete_path)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  // ---------------- Insert phase ----------------
  FILE* fptr = fopen(insert_path.c_str(), "rb");
  if (!fptr) {
    perror(("Unable to open file: " + insert_path).c_str());
    return 0.0f;
  }

  uint32_t* data = new uint32_t[num_ins];

  if (fread(data, sizeof(uint32_t), num_ins, fptr) != num_ins) {
    perror(("Unable to read file: " + insert_path).c_str());
    fclose(fptr);
    delete[] data;
    return 0.0f;
  }
  fclose(fptr);

  uint32_t batchSize = 100000000;

  cuco::pair<Key, Value>* query_pairs_begin{};
  cudaMallocManaged(&query_pairs_begin,
                    sizeof(cuco::pair<Key, Value>) * batchSize);

  uint64_t cpu_counter = 0;
  std::mt19937 mt(32);
  std::uniform_int_distribution<uint32_t> valueDistribution(1, UINT32_MAX - 1);

  while (cpu_counter < num_ins) {
    uint32_t per_batch_gpu_ins = 0;
    while (per_batch_gpu_ins < batchSize && cpu_counter < num_ins) {
      query_pairs_begin[per_batch_gpu_ins].first  = data[cpu_counter];
      query_pairs_begin[per_batch_gpu_ins].second = valueDistribution(mt);
      ++per_batch_gpu_ins;
      ++cpu_counter;
    }

    cudaMemAdvise(query_pairs_begin,
                  per_batch_gpu_ins * sizeof(cuco::pair<Key, Value>),
                  cudaMemAdviseSetAccessedBy,
                  0);
    cudaMemPrefetchAsync(query_pairs_begin,
                         per_batch_gpu_ins * sizeof(cuco::pair<Key, Value>),
                         0);

    map.insert_or_assign(query_pairs_begin,
                         query_pairs_begin + per_batch_gpu_ins);
  }

  delete[] data;
  cudaFree(query_pairs_begin);

  // ---------------- Delete phase ----------------
  FILE* fptr_delete = fopen(delete_path.c_str(), "rb");
  if (!fptr_delete) {
    perror(("Unable to open file: " + delete_path).c_str());
    return 0.0f;
  }

  uint32_t* data_delete = new uint32_t[num_keys];

  if (fread(data_delete, sizeof(uint32_t), num_keys, fptr_delete) != num_keys) {
    perror(("Unable to read file: " + delete_path).c_str());
    fclose(fptr_delete);
    delete[] data_delete;
    return 0.0f;
  }
  fclose(fptr_delete);

  Key* keys_begin{};
  cudaMallocManaged(&keys_begin, sizeof(Key) * batchSize);

  cpu_counter = 0;
  float total_delete_time = 0.0f;

  while (cpu_counter < num_keys) {
    uint32_t per_batch_gpu_del = 0;
    while (per_batch_gpu_del < batchSize && cpu_counter < num_keys) {
      keys_begin[per_batch_gpu_del] = data_delete[cpu_counter];
      ++per_batch_gpu_del;
      ++cpu_counter;
    }

    cudaMemAdvise(keys_begin,
                  per_batch_gpu_del * sizeof(Key),
                  cudaMemAdviseSetAccessedBy,
                  0);
    cudaMemPrefetchAsync(keys_begin,
                         per_batch_gpu_del * sizeof(Key),
                         0);

    auto start = HRClock::now();
    map.erase(keys_begin,
             keys_begin + per_batch_gpu_del);
    auto end = HRClock::now();

    total_delete_time += DurationFloatMS(end - start).count();
  }

  std::cout << "Total delete time: " << total_delete_time << " ms\n";

  delete[] data_delete;
  cudaFree(keys_begin);

  return total_delete_time;
}


// -----------------------------------------------------------------------------
// Catch2 Test
// -----------------------------------------------------------------------------
TEMPLATE_TEST_CASE_SIG(
  "static_map: delete tests",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe, int CGSize),
   Key,
   Value,
   Probe,
   CGSize),
  (uint32_t, uint32_t, cuco::test::probe_sequence::double_hashing, 16))
{
  constexpr size_type num_ins = 4000000000;
  size_type num_keys = g_num_keys;
  int runs = 2;

  float total_delete_time = 0.0f;

  for (int i = 0; i < runs; ++i) {
    using probe =
      std::conditional_t<
        Probe == cuco::test::probe_sequence::linear_probing,
        cuco::linear_probing<CGSize, cuco::xxhash_32<Key>>,
        cuco::double_hashing<CGSize,
                             cuco::xxhash_32<Key>,
                             cuco::xxhash_32<Key>>>;

    auto map =
      cuco::static_map<Key,
                       Value,
                       cuco::extent<size_type>,
                       cuda::thread_scope_device,
                       cuda::std::equal_to<Key>,
                       probe,
                       cuco::cuda_allocator<cuda::std::byte>,
                       cuco::storage<2>>{
        num_ins,
        cuco::empty_key<Key>{0},
        cuco::empty_value<Value>{0}};

    std::printf("Calling to CG size : %d\n", CGSize);

    total_delete_time += test_unique_sequence(
      map,
      num_keys,
      num_ins,
      g_insert_path,
      g_delete_path);
  }

  std::cout << "Average delete time for " << num_keys
            << " keys (ms): "
            << (total_delete_time / runs) << "\n";
  std::cout << "---------------------------------\n";
}


// -----------------------------------------------------------------------------
// Custom Catch2 main() with CLI options
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  Catch::Session session;

  using namespace Catch::Clara;

  std::string insert_path;
  std::string delete_path;
  size_type num_keys = 0;

  auto cli =
    session.cli()
    | Opt(insert_path, "path")["--insert-path"]("Insert trace file path")
    | Opt(delete_path, "path")["--delete-path"]("Delete trace file path")
    | Opt(num_keys, "count")["--num-keys"]("Number of keys to delete");

  session.cli(cli);

  int rc = session.applyCommandLine(argc, argv);
  if (rc != 0) return rc;

  if (!insert_path.empty()) g_insert_path = insert_path;
  if (!delete_path.empty())   g_delete_path   = delete_path;
  if (num_keys != 0)        g_num_keys     = num_keys;

  return session.run();
}
