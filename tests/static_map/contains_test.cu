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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

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

template <typename Map>
float test_unique_sequence(Map& map, size_type num_keys, size_t num_ins)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

string absolute_path =
    "/data/heterods-trace/insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE_5e8_Random_Clusters.bin";

  // Open file in binary read mode
  FILE* fptr = fopen(absolute_path.c_str(), "rb");
  if (!fptr) {
    string error_msg = "Unable to open file: " + absolute_path;
    perror(error_msg.c_str());
    return 0;
  }

  // Suppose you know the number of uint32_t elements to read;  // replace with actual number
  uint32_t* data = new uint32_t[num_ins];

  size_t freadStatus = fread(data, sizeof(uint32_t), num_ins, fptr);
  if (freadStatus != num_ins) {  // check if all elements were read
    string error_string = "Unable to read the file " + absolute_path;
    perror(error_string.c_str());
    fclose(fptr);
    delete[] data;
    return 0;
  }

  uint32_t batchSize = 100000000;
  cuco::pair<Key, Value>* query_pairs_begin;
  cudaMallocManaged(&query_pairs_begin, sizeof(cuco::pair<Key, Value>) * batchSize);
  float total_search_time = 0.0f;
  uint64_t cpu_counter    = 0;
  std::mt19937 mt(32);
  std::uniform_int_distribution<uint32_t> valueDistribution(1, UINT32_MAX - 1);

  while (cpu_counter < num_ins) {
    uint32_t per_batch_gpu_ins = 0;
    while (per_batch_gpu_ins < batchSize && cpu_counter < num_ins) {
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

    map.insert_or_assign(query_pairs_begin,query_pairs_begin + per_batch_gpu_ins);
  }
  cudaFree(query_pairs_begin);
  Key* keys_begin;
  cudaMallocManaged(&keys_begin, sizeof(Key) * batchSize);
  bool* result;
  cudaMallocManaged(&result,sizeof(bool)*batchSize);

  cpu_counter = 0;
  absolute_path =
    "/data/heterods-trace/insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin";

  // Open file in binary read mode
  FILE* fptr_del = fopen(absolute_path.c_str(), "rb");
  if (!fptr_del) {
    string error_msg = "Unable to open file: " + absolute_path;
    perror(error_msg.c_str());
    return 0;
  }

  // Suppose you know the number of uint32_t elements to read;  // replace with actual number
  uint32_t* data_del = new uint32_t[num_keys];

  size_t freadStatus_del = fread(data_del, sizeof(uint32_t), num_keys, fptr_del);
  if (freadStatus_del != num_keys) {  // check if all elements were read
    string error_string = "Unable to read the file " + absolute_path;
    perror(error_string.c_str());
    fclose(fptr_del);
    delete[] data_del;
    return 0;
  }
  
  while (cpu_counter < num_keys) {
    uint32_t per_batch_gpu_find = 0;
    while (per_batch_gpu_find < batchSize && cpu_counter < num_keys) {
      keys_begin[per_batch_gpu_find] = data[cpu_counter];
      per_batch_gpu_find++;
      cpu_counter++;
    }
    cudaMemAdvise(
      keys_begin, (per_batch_gpu_find * sizeof(Key)), cudaMemAdviseSetAccessedBy, 0);
    cudaMemPrefetchAsync(keys_begin, (per_batch_gpu_find * sizeof(Key)), 0);
    auto start = HRClock::now();
    map.contains(keys_begin, keys_begin + per_batch_gpu_find,result);
    auto end                 = HRClock::now();
    DurationFloatMS duration = end - start;
    total_search_time += duration.count();
  }
  std::cout << "Total search time: " << total_search_time << "\n";
  cudaFree(keys_begin);
  return total_search_time;
}

  

TEMPLATE_TEST_CASE_SIG(
  "static_map: contains + retrieve_all tests",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe, int CGSize),
   Key,
   Value,
   Probe,
   CGSize),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 16))
{
  size_type num_keys{2000000000};
  constexpr size_type num_ins{4000000000};
  int runs = 2;
  // while (num_keys <= 4000000000) {
    float total_search_time = 0.0f;
    for (int i = 0; i < runs; i++) {
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
    num_ins, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}, cuco::erased_key<Key>{-2}};

      printf("Calling to CG size : %d\n", CGSize);
      total_search_time += test_unique_sequence(map, num_keys, num_ins);
    }
    std::cout << "Total time taken for search for " << num_keys
              << " elements (ms): " << (total_search_time / runs) << "\n";
    std::cout << "-----------------------------\n";
    // num_keys += 500000000;
  // }
}
