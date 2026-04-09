/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#pragma once

#include <cuco/detail/hash_functions/utils.cuh>
#include <cuco/extent.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <type_traits>
#include <cstdint>

namespace cuco::detail {

template <typename Key>
struct fibHash32 {
 private:
  static constexpr std::uint64_t prime1 = 11400714819323198485ull;

 public:
  using argument_type = Key;
  using result_type   = std::uint64_t;

  __host__ __device__ constexpr fibHash32(std::uint32_t seed = 0) : seed_{seed} {}

  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    static_assert(std::is_integral_v<Key> || std::is_enum_v<Key>,
                  "fibHash32 requires Key to be an integral or enum type");

    std::uint64_t k = static_cast<std::uint64_t>(key);
    std::uint64_t h = k * prime1;

    return finalize(static_cast<std::uint32_t>(h));
  }

 private:
  constexpr __host__ __device__
  std::uint32_t finalize(std::uint32_t h) const noexcept
  {
    return h;
  }

  std::uint32_t seed_;
};

}  // namespace cuco::detail
