/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuco/hash_functions.cuh>
#include <cuco/static_map.cuh>

#include <cuda/functional>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

#include <catch2/catch_session.hpp>
#include <catch2/catch_template_test_macros.hpp>

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

static string g_insert_path =
  "/data/heterods-trace/insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE_5e8_Random_Clusters.bin";

static size_type g_num_keys = 2000000000;

template <typename Map>
float test_insert_or_assign(Map& map, size_type num_keys, const std::string& insert_path)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  // Open file in binary read mode
  FILE* fptr = fopen(insert_path.c_str(), "rb");
  if (!fptr) {
    perror(("Unable to open file: " + insert_path).c_str());
    return 0.0f;
  }

  // Suppose you know the number of uint32_t elements to read;  // replace with actual number
  uint32_t* data = new uint32_t[num_keys];

  if (fread(data, sizeof(uint32_t), num_keys, fptr) != num_keys) {
    perror(("Unable to read file: " + insert_path).c_str());
    fclose(fptr);
    delete[] data;
    return 0.0f;
  }
  fclose(fptr);

  uint32_t batchSize = 100000000;
  cuco::pair<Key, Value>* query_pairs_begin;
  printf("Size of pair: %lu\n", sizeof(cuco::pair<Key, Value>));
  // cuco::pair<Key, Value>* h_query_pairs_begin = new cuco::pair<Key, Value>[batchSize];
  cudaMallocManaged(&query_pairs_begin, sizeof(cuco::pair<Key, Value>) * batchSize);
  float total_insert_time = 0.0f;
  uint64_t cpu_counter    = 0;
  std::mt19937 mt(32);
  std::uniform_int_distribution<uint32_t> valueDistribution(1, UINT32_MAX - 1);
  while (cpu_counter < num_keys) {
    uint32_t per_batch_gpu_ins = 0;
    while (per_batch_gpu_ins < batchSize && cpu_counter < num_keys) {
      query_pairs_begin[per_batch_gpu_ins].first  = data[cpu_counter];
      query_pairs_begin[per_batch_gpu_ins].second = valueDistribution(mt);
      per_batch_gpu_ins++;
      cpu_counter++;
    }
    cudaMemAdvise(query_pairs_begin,
                  (per_batch_gpu_ins * sizeof(cuco::pair<Key, Value>)),
                  cudaMemAdviseSetAccessedBy,
                  0);
    cudaMemPrefetchAsync(
      query_pairs_begin, (per_batch_gpu_ins * sizeof(cuco::pair<Key, Value>)), 0);
    // cudaMemcpy(query_pairs_begin,
    //            h_query_pairs_begin,
    //            sizeof(cuco::pair<Key, Value>) * batchSize,
    //            cudaMemcpyHostToDevice);
    auto start = HRClock::now();
    map.insert_or_assign(query_pairs_begin, query_pairs_begin + per_batch_gpu_ins);
    auto end                 = HRClock::now();
    DurationFloatMS duration = end - start;
    total_insert_time += duration.count();
  }
  cudaFree(query_pairs_begin);
  std::cout << "Insert time for all batches: " << total_insert_time << "\n";

  // uint32_t* keys;
  // cudaMallocManaged(&keys, sizeof(uint32_t) * num_keys);
  // uint32_t* values;
  // cudaMallocManaged(&values, sizeof(uint32_t) * num_keys);
  // map.retrieve_all(keys, values);
  // // for (uint32_t j = 0; j < 10; j++) {
  // // printf("keys: %u\n", keys[j]);
  // // }
  // std::sort(keys, keys + num_keys);
  // std::sort(data, data + num_keys);
  // uint64_t count = 0;
  // for (uint32_t j = 0; j < num_keys; j++) {
  //   if (keys[j] != data[j]) { count++; }
  // }
  // printf("Total mismatches: %lu\n", count);
  return total_insert_time;
}

TEMPLATE_TEST_CASE_SIG(
  "static_map insert_or_assign tests",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe, int CGSize),
   Key,
   Value,
   Probe,
   CGSize),
  (uint32_t, uint32_t, cuco::test::probe_sequence::double_hashing, 16))
{
  int runs = 1;
  size_type num_keys{g_num_keys};
  // while (num_keys <= 4000000000) {
  float total_insert_time = 0.0f;
  for (int i = 0; i < 1; i++) {
    using probe =
      std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                         cuco::linear_probing<CGSize, cuco::xxhash_32<Key>>,
                         cuco::double_hashing<CGSize, cuco::xxhash_32<Key>, cuco::xxhash_32<Key>>>;

    auto map = cuco::static_map<Key,
                                Value,
                                cuco::extent<size_type>,
                                cuda::thread_scope_device,
                                cuda::std::equal_to<Key>,
                                probe,
                                cuco::cuda_allocator<cuda::std::byte>,
                                cuco::storage<2>>{
      (num_keys), cuco::empty_key<Key>{0}, cuco::empty_value<Value>{0}};

    printf("Calling to CG size : %d\n", CGSize);
    total_insert_time += test_insert_or_assign(map, num_keys, g_insert_path);
  }

  std::cout << "Total time taken for insert for " << num_keys
            << " elements (ms): " << (total_insert_time / runs) << "\n";
  std::cout << "-----------------------------\n";
  // num_keys += 500000000;
  // }
}

int main(int argc, char* argv[])
{
  Catch::Session session;

  using namespace Catch::Clara;

  std::string insert_path;
  size_type num_keys = 0;

  auto cli = session.cli() | Opt(insert_path, "path")["--insert-path"]("Insert trace file path") |
             Opt(num_keys, "count")["--num-keys"]("Number of keys to find");

  session.cli(cli);

  int rc = session.applyCommandLine(argc, argv);
  if (rc != 0) return rc;

  if (!insert_path.empty()) g_insert_path = insert_path;
  if (num_keys != 0) g_num_keys = num_keys;

  return session.run();
}
