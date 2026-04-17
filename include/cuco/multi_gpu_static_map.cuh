#pragma once
#include <cuco/detail/error.hpp>
#include <cuco/detail/multi_gpu/partition_kernels.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/pair.cuh>
#include <cuco/static_map.cuh>
#include <cuco/utility/allocator.hpp>
#include <cuda_runtime.h>
#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace cuco {

template <class Key, class T,
          class ProbingScheme = cuco::double_hashing<4, cuco::xxhash_32<Key>, cuco::xxhash_32<Key>>,
          class Storage = cuco::storage<2>>
class multi_gpu_static_map {
 public:
  using key_type = Key; using mapped_type = T; using value_type = cuco::pair<Key, T>; using size_type = std::size_t;
  using per_gpu_map_type = cuco::static_map<Key, T, cuco::extent<size_type>, cuda::thread_scope_device,
    cuda::std::equal_to<Key>, ProbingScheme, cuco::cuda_allocator<cuda::std::byte>, Storage>;

  multi_gpu_static_map(size_type total_capacity, std::vector<int> const& gpu_ids,
                       cuco::empty_key<Key> empty_key, cuco::empty_value<T> empty_value)
    : gpu_ids_(gpu_ids), num_gpus_(gpu_ids.size()), empty_key_(empty_key.value),
      empty_value_(empty_value.value), staging_gpu_(gpu_ids[0])
  {
    assert(num_gpus_ > 0);
    size_type per_gpu = (total_capacity + num_gpus_ - 1) / num_gpus_;
    int orig; CUCO_CUDA_TRY(cudaGetDevice(&orig));
    streams_.resize(num_gpus_);
    for (int i = 0; i < num_gpus_; ++i) {
      CUCO_CUDA_TRY(cudaSetDevice(gpu_ids_[i]));
      CUCO_CUDA_TRY(cudaStreamCreate(&streams_[i]));
      maps_.emplace_back(std::make_unique<per_gpu_map_type>(per_gpu, cuco::empty_key<Key>{empty_key_}, cuco::empty_value<T>{empty_value_}));
    }
    CUCO_CUDA_TRY(cudaSetDevice(orig));
    enable_peer_access_();
  }

  ~multi_gpu_static_map() { for (auto& s : streams_) if (s) cudaStreamDestroy(s); }
  multi_gpu_static_map(multi_gpu_static_map const&) = delete;
  multi_gpu_static_map& operator=(multi_gpu_static_map const&) = delete;

  template <typename InputIt> void insert(InputIt first, InputIt last) {
    auto n = static_cast<size_type>(last - first); if (n == 0) return;
    auto p = partition_pairs_(first, n);
    int orig; CUCO_CUDA_TRY(cudaGetDevice(&orig));
    for (int i = 0; i < num_gpus_; ++i) {
      if (p.counts[i] == 0) continue;
      CUCO_CUDA_TRY(cudaSetDevice(gpu_ids_[i]));
      apply_uvm_hints_(p.buffers[i], p.counts[i] * sizeof(value_type), gpu_ids_[i], streams_[i]);
      maps_[i]->insert_async(p.buffers[i], p.buffers[i] + p.counts[i], cuda::stream_ref{streams_[i]});
    }
    sync_all_(); CUCO_CUDA_TRY(cudaSetDevice(orig)); free_pp_(p);
  }

  template <typename InputIt> void insert_or_assign(InputIt first, InputIt last) {
    auto n = static_cast<size_type>(last - first); if (n == 0) return;
    auto p = partition_pairs_(first, n);
    int orig; CUCO_CUDA_TRY(cudaGetDevice(&orig));
    for (int i = 0; i < num_gpus_; ++i) {
      if (p.counts[i] == 0) continue;
      CUCO_CUDA_TRY(cudaSetDevice(gpu_ids_[i]));
      apply_uvm_hints_(p.buffers[i], p.counts[i] * sizeof(value_type), gpu_ids_[i], streams_[i]);
      maps_[i]->insert_or_assign_async(p.buffers[i], p.buffers[i] + p.counts[i], cuda::stream_ref{streams_[i]});
    }
    sync_all_(); CUCO_CUDA_TRY(cudaSetDevice(orig)); free_pp_(p);
  }

  template <typename InputIt, typename OutputIt> void find(InputIt first, InputIt last, OutputIt output_begin) {
    auto n = static_cast<size_type>(last - first); if (n == 0) return;
    auto p = partition_keys_(first, n);
    int orig; CUCO_CUDA_TRY(cudaGetDevice(&orig));
    std::vector<mapped_type*> rbufs(num_gpus_, nullptr);
    for (int i = 0; i < num_gpus_; ++i) {
      if (p.counts[i] == 0) continue;
      CUCO_CUDA_TRY(cudaMallocManaged(&rbufs[i], sizeof(mapped_type) * p.counts[i]));
    }
    for (int i = 0; i < num_gpus_; ++i) {
      if (p.counts[i] == 0) continue;
      CUCO_CUDA_TRY(cudaSetDevice(gpu_ids_[i]));
      apply_uvm_hints_(p.key_buffers[i], p.counts[i] * sizeof(key_type), gpu_ids_[i], streams_[i]);
      apply_uvm_hints_(rbufs[i], p.counts[i] * sizeof(mapped_type), gpu_ids_[i], streams_[i]);
      maps_[i]->find_async(p.key_buffers[i], p.key_buffers[i] + p.counts[i], rbufs[i], cuda::stream_ref{streams_[i]});
    }
    sync_all_();
    CUCO_CUDA_TRY(cudaSetDevice(staging_gpu_));
    for (int i = 0; i < num_gpus_; ++i) {
      if (p.counts[i] == 0) continue;
      auto g = detail::multi_gpu::grid_size_for(p.counts[i]);
      detail::multi_gpu::gather_results_kernel<<<g, 256>>>(rbufs[i], p.index_maps[i], p.counts[i], output_begin);
    }
    CUCO_CUDA_TRY(cudaDeviceSynchronize()); CUCO_CUDA_TRY(cudaSetDevice(orig));
    for (int i = 0; i < num_gpus_; ++i) if (rbufs[i]) CUCO_CUDA_TRY(cudaFree(rbufs[i]));
    free_kp_(p);
  }

  template <typename InputIt, typename OutputIt> void contains(InputIt first, InputIt last, OutputIt output_begin) {
    auto n = static_cast<size_type>(last - first); if (n == 0) return;
    auto p = partition_keys_(first, n);
    int orig; CUCO_CUDA_TRY(cudaGetDevice(&orig));
    std::vector<bool*> rbufs(num_gpus_, nullptr);
    for (int i = 0; i < num_gpus_; ++i) {
      if (p.counts[i] == 0) continue;
      CUCO_CUDA_TRY(cudaMallocManaged(&rbufs[i], sizeof(bool) * p.counts[i]));
    }
    for (int i = 0; i < num_gpus_; ++i) {
      if (p.counts[i] == 0) continue;
      CUCO_CUDA_TRY(cudaSetDevice(gpu_ids_[i]));
      apply_uvm_hints_(p.key_buffers[i], p.counts[i] * sizeof(key_type), gpu_ids_[i], streams_[i]);
      apply_uvm_hints_(rbufs[i], p.counts[i] * sizeof(bool), gpu_ids_[i], streams_[i]);
      maps_[i]->contains_async(p.key_buffers[i], p.key_buffers[i] + p.counts[i], rbufs[i], cuda::stream_ref{streams_[i]});
    }
    sync_all_();
    CUCO_CUDA_TRY(cudaSetDevice(staging_gpu_));
    for (int i = 0; i < num_gpus_; ++i) {
      if (p.counts[i] == 0) continue;
      auto g = detail::multi_gpu::grid_size_for(p.counts[i]);
      detail::multi_gpu::gather_results_kernel<<<g, 256>>>(rbufs[i], p.index_maps[i], p.counts[i], output_begin);
    }
    CUCO_CUDA_TRY(cudaDeviceSynchronize()); CUCO_CUDA_TRY(cudaSetDevice(orig));
    for (int i = 0; i < num_gpus_; ++i) if (rbufs[i]) CUCO_CUDA_TRY(cudaFree(rbufs[i]));
    free_kp_(p);
  }

  template <typename InputIt> void erase(InputIt first, InputIt last) {
    auto n = static_cast<size_type>(last - first); if (n == 0) return;
    auto p = partition_keys_(first, n);
    int orig; CUCO_CUDA_TRY(cudaGetDevice(&orig));
    for (int i = 0; i < num_gpus_; ++i) {
      if (p.counts[i] == 0) continue;
      CUCO_CUDA_TRY(cudaSetDevice(gpu_ids_[i]));
      apply_uvm_hints_(p.key_buffers[i], p.counts[i] * sizeof(key_type), gpu_ids_[i], streams_[i]);
      maps_[i]->erase_async(p.key_buffers[i], p.key_buffers[i] + p.counts[i], cuda::stream_ref{streams_[i]});
    }
    sync_all_(); CUCO_CUDA_TRY(cudaSetDevice(orig)); free_kp_(p);
  }

  void clear() {
    int orig; CUCO_CUDA_TRY(cudaGetDevice(&orig));
    for (int i = 0; i < num_gpus_; ++i) { CUCO_CUDA_TRY(cudaSetDevice(gpu_ids_[i])); maps_[i]->clear(cuda::stream_ref{streams_[i]}); }
    sync_all_(); CUCO_CUDA_TRY(cudaSetDevice(orig));
  }

  size_type size() const {
    int orig; CUCO_CUDA_TRY(cudaGetDevice(&orig)); size_type t = 0;
    for (int i = 0; i < num_gpus_; ++i) { CUCO_CUDA_TRY(cudaSetDevice(gpu_ids_[i])); t += maps_[i]->size(cuda::stream_ref{streams_[i]}); }
    CUCO_CUDA_TRY(cudaSetDevice(orig)); return t;
  }

  std::vector<size_type> per_gpu_sizes() const {
    int orig; CUCO_CUDA_TRY(cudaGetDevice(&orig)); std::vector<size_type> s(num_gpus_);
    for (int i = 0; i < num_gpus_; ++i) { CUCO_CUDA_TRY(cudaSetDevice(gpu_ids_[i])); s[i] = maps_[i]->size(cuda::stream_ref{streams_[i]}); }
    CUCO_CUDA_TRY(cudaSetDevice(orig)); return s;
  }

  int num_gpus() const noexcept { return num_gpus_; }

 private:
  struct pair_partitions { std::vector<value_type*> buffers; std::vector<size_type> counts; };
  struct key_partitions { std::vector<key_type*> key_buffers; std::vector<size_type*> index_maps; std::vector<size_type> counts; };

  template <typename InputIt> pair_partitions partition_pairs_(InputIt first, size_type n) {
    int orig; CUCO_CUDA_TRY(cudaGetDevice(&orig)); CUCO_CUDA_TRY(cudaSetDevice(staging_gpu_));
    pair_partitions r; r.counts.resize(num_gpus_, 0); r.buffers.resize(num_gpus_, nullptr);
    size_type* dc = nullptr; CUCO_CUDA_TRY(cudaMallocManaged(&dc, sizeof(size_type) * num_gpus_));
    CUCO_CUDA_TRY(cudaMemset(dc, 0, sizeof(size_type) * num_gpus_));
    auto g = detail::multi_gpu::grid_size_for(n);
    detail::multi_gpu::count_per_partition_kernel<<<g, 256>>>(first, n, num_gpus_, dc);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
    for (int i = 0; i < num_gpus_; ++i) { r.counts[i] = dc[i]; if (r.counts[i] > 0) CUCO_CUDA_TRY(cudaMallocManaged(&r.buffers[i], sizeof(value_type) * r.counts[i])); }
    value_type** dp = nullptr; CUCO_CUDA_TRY(cudaMallocManaged(&dp, sizeof(value_type*) * num_gpus_));
    for (int i = 0; i < num_gpus_; ++i) dp[i] = r.buffers[i];
    size_type* dw = nullptr; CUCO_CUDA_TRY(cudaMallocManaged(&dw, sizeof(size_type) * num_gpus_));
    CUCO_CUDA_TRY(cudaMemset(dw, 0, sizeof(size_type) * num_gpus_));
    detail::multi_gpu::scatter_pairs_kernel<<<g, 256>>>(first, n, num_gpus_, dp, dw);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
    CUCO_CUDA_TRY(cudaFree(dc)); CUCO_CUDA_TRY(cudaFree(dp)); CUCO_CUDA_TRY(cudaFree(dw));
    CUCO_CUDA_TRY(cudaSetDevice(orig)); return r;
  }

  template <typename InputIt> key_partitions partition_keys_(InputIt first, size_type n) {
    int orig; CUCO_CUDA_TRY(cudaGetDevice(&orig)); CUCO_CUDA_TRY(cudaSetDevice(staging_gpu_));
    key_partitions r; r.counts.resize(num_gpus_, 0); r.key_buffers.resize(num_gpus_, nullptr); r.index_maps.resize(num_gpus_, nullptr);
    size_type* dc = nullptr; CUCO_CUDA_TRY(cudaMallocManaged(&dc, sizeof(size_type) * num_gpus_));
    CUCO_CUDA_TRY(cudaMemset(dc, 0, sizeof(size_type) * num_gpus_));
    auto g = detail::multi_gpu::grid_size_for(n);
    detail::multi_gpu::count_keys_per_partition_kernel<<<g, 256>>>(first, n, num_gpus_, dc);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
    for (int i = 0; i < num_gpus_; ++i) { r.counts[i] = dc[i]; if (r.counts[i] > 0) { CUCO_CUDA_TRY(cudaMallocManaged(&r.key_buffers[i], sizeof(key_type)*r.counts[i])); CUCO_CUDA_TRY(cudaMallocManaged(&r.index_maps[i], sizeof(size_type)*r.counts[i])); } }
    key_type** dkp = nullptr; CUCO_CUDA_TRY(cudaMallocManaged(&dkp, sizeof(key_type*) * num_gpus_));
    size_type** dip = nullptr; CUCO_CUDA_TRY(cudaMallocManaged(&dip, sizeof(size_type*) * num_gpus_));
    for (int i = 0; i < num_gpus_; ++i) { dkp[i] = r.key_buffers[i]; dip[i] = r.index_maps[i]; }
    size_type* dw = nullptr; CUCO_CUDA_TRY(cudaMallocManaged(&dw, sizeof(size_type) * num_gpus_));
    CUCO_CUDA_TRY(cudaMemset(dw, 0, sizeof(size_type) * num_gpus_));
    detail::multi_gpu::scatter_keys_kernel<<<g, 256>>>(first, n, num_gpus_, dkp, dip, dw);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
    CUCO_CUDA_TRY(cudaFree(dc)); CUCO_CUDA_TRY(cudaFree(dkp)); CUCO_CUDA_TRY(cudaFree(dip)); CUCO_CUDA_TRY(cudaFree(dw));
    CUCO_CUDA_TRY(cudaSetDevice(orig)); return r;
  }

  void free_pp_(pair_partitions& p) { for (int i = 0; i < num_gpus_; ++i) if (p.buffers[i]) CUCO_CUDA_TRY(cudaFree(p.buffers[i])); }
  void free_kp_(key_partitions& p) { for (int i = 0; i < num_gpus_; ++i) { if (p.key_buffers[i]) CUCO_CUDA_TRY(cudaFree(p.key_buffers[i])); if (p.index_maps[i]) CUCO_CUDA_TRY(cudaFree(p.index_maps[i])); } }

  void apply_uvm_hints_(void* ptr, size_t bytes, int dev, cudaStream_t stream = 0) {
    if (!ptr || bytes == 0) return;
    cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, dev);
    cudaMemPrefetchAsync(ptr, bytes, dev, stream);
  }

  void sync_all_() { for (int i = 0; i < num_gpus_; ++i) CUCO_CUDA_TRY(cudaStreamSynchronize(streams_[i])); }

  void enable_peer_access_() {
    for (int i = 0; i < num_gpus_; ++i) for (int j = 0; j < num_gpus_; ++j) {
      if (i == j) continue; int ok = 0;
      CUCO_CUDA_TRY(cudaDeviceCanAccessPeer(&ok, gpu_ids_[i], gpu_ids_[j]));
      if (ok) { CUCO_CUDA_TRY(cudaSetDevice(gpu_ids_[i])); auto e = cudaDeviceEnablePeerAccess(gpu_ids_[j], 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CUCO_CUDA_TRY(e);
        if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); }
    }
  }

  std::vector<int> gpu_ids_; int num_gpus_; key_type empty_key_; mapped_type empty_value_; int staging_gpu_;
  std::vector<std::unique_ptr<per_gpu_map_type>> maps_; std::vector<cudaStream_t> streams_;
};
}
