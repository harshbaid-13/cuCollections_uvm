#include <cuco/multi_gpu_static_map.cuh>
#include <cuco/pair.cuh>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Key = int32_t; using Value = int32_t; using pair_type = cuco::pair<Key, Value>;

__global__ void gen_pairs(pair_type* p, std::size_t n) {
  auto tid = blockIdx.x*(std::size_t)blockDim.x+threadIdx.x; auto s = gridDim.x*(std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s) { p[i].first = (Key)(i+1); p[i].second = (Key)(i+1)*10; }
}
__global__ void get_keys(pair_type const* p, Key* k, std::size_t n) {
  auto tid = blockIdx.x*(std::size_t)blockDim.x+threadIdx.x; auto s = gridDim.x*(std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s) k[i] = p[i].first;
}
__global__ void verify(Key const* k, Value const* v, std::size_t n, uint64_t* err, Value mul) {
  auto tid = blockIdx.x*(std::size_t)blockDim.x+threadIdx.x; auto s = gridDim.x*(std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s) if (v[i] != k[i]*mul) atomicAdd((unsigned long long*)err, 1ULL);
}
__global__ void count_true(bool const* f, std::size_t n, uint64_t* c) {
  auto tid = blockIdx.x*(std::size_t)blockDim.x+threadIdx.x; auto s = gridDim.x*(std::size_t)blockDim.x;
  for (std::size_t i = tid; i < n; i += s) if (f[i]) atomicAdd((unsigned long long*)c, 1ULL);
}

struct Timer { using C=std::chrono::high_resolution_clock; C::time_point t; Timer():t(C::now()){} float ms(){return std::chrono::duration<float,std::milli>(C::now()-t).count();} };
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

int main(int argc, char** argv) {
  std::size_t N = argc>1 ? std::atol(argv[1]) : 100000000ULL;
  constexpr int ngpus = 3;
  std::vector<int> const gpus{1, 2, 3};
  int dev_count; CK(cudaGetDeviceCount(&dev_count));
  if (dev_count < 1 + ngpus) {
    fprintf(stderr, "Need at least %d CUDA devices (GPU 0 unused; using IDs 1..%d), found %d\n",
            1 + ngpus, ngpus, dev_count);
    return 1;
  }
  int const host_dev = gpus[0];
  CK(cudaSetDevice(host_dev));

  printf("╔═══════════════════════════════════════════════════════╗\n");
  printf("║   Multi-GPU Hash Map Test (WarpDrive + UVM)          ║\n");
  printf("╠═══════════════════════════════════════════════════════╣\n");
  printf("║  GPUs:           %-3d (devices 1,2,3; not 0)          ║\n", ngpus);
  for (int i = 0; i < ngpus; i++) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, gpus[i]));
    printf("║    GPU %d: %-24s %4.1f GB     ║\n", gpus[i], p.name, p.totalGlobalMem/(1024.*1024.*1024.));
  }
  printf("║  Elements:       %-12zu                       ║\n", N);
  printf("║  Data size:      %-5.2f GB                             ║\n", N*8/(1024.*1024.*1024.));
  printf("╚═══════════════════════════════════════════════════════╝\n\n");

  std::size_t cap = (std::size_t)(N / 0.8);
  printf("[1] Creating multi-GPU map (capacity %zu)...\n", cap);
  Timer tc;
  auto map = cuco::multi_gpu_static_map<Key,Value>(cap, gpus, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1});
  printf("    Done in %.1f ms\n\n", tc.ms());

  printf("[2] Generating %zu pairs...\n", N);
  pair_type* d_pairs; CK(cudaMallocManaged(&d_pairs, sizeof(pair_type)*N));
  for (int d : gpus)
    cudaMemAdvise(d_pairs, sizeof(pair_type)*N, cudaMemAdviseSetAccessedBy, d);
  int bl=256, gr=std::min((int)((N+bl-1)/bl),65535);
  gen_pairs<<<gr,bl>>>(d_pairs,N); CK(cudaDeviceSynchronize()); printf("    Done.\n\n");

  printf("[3] Insert...\n"); Timer ti;
  map.insert(d_pairs, d_pairs+N); float ins=ti.ms();
  printf("    %.1f ms (%.2f M ops/s)\n", ins, N/1e6/(ins/1e3));
  auto sizes = map.per_gpu_sizes(); std::size_t tot=0;
  for (int i = 0; i < ngpus; i++) {
    printf("    GPU %d: %zu elements\n", gpus[i], sizes[i]);
    tot += sizes[i];
  }
  printf("    Total: %zu\n\n", tot);

  printf("[4] Find...\n");
  Key* d_keys; CK(cudaMallocManaged(&d_keys, sizeof(Key)*N));
  get_keys<<<gr,bl>>>(d_pairs,d_keys,N); CK(cudaDeviceSynchronize());
  Value* d_vals; CK(cudaMallocManaged(&d_vals, sizeof(Value)*N));
  Timer tf; map.find(d_keys, d_keys+N, d_vals); float fnd=tf.ms();
  printf("    %.1f ms (%.2f M ops/s)\n", fnd, N/1e6/(fnd/1e3));
  uint64_t* d_err; CK(cudaMallocManaged(&d_err, 8)); *d_err=0;
  verify<<<gr,bl>>>(d_keys,d_vals,N,d_err,10); CK(cudaDeviceSynchronize());
  printf("    %s\n\n", *d_err==0 ? "✓ PASSED" : "✗ FAILED");

  printf("[5] Contains...\n");
  bool* d_found; CK(cudaMallocManaged(&d_found, sizeof(bool)*N));
  Timer tcon; map.contains(d_keys, d_keys+N, d_found); float con=tcon.ms();
  printf("    %.1f ms (%.2f M ops/s)\n", con, N/1e6/(con/1e3));
  uint64_t* d_cnt; CK(cudaMallocManaged(&d_cnt, 8)); *d_cnt=0;
  count_true<<<gr,bl>>>(d_found,N,d_cnt); CK(cudaDeviceSynchronize());
  printf("    %s (%lu/%zu)\n\n", *d_cnt==N ? "✓ PASSED" : "✗ FAILED", *d_cnt, N);

  printf("[6] Insert-or-assign...\n"); Timer tu;
  map.insert_or_assign(d_pairs, d_pairs+N); float ups=tu.ms();
  printf("    %.1f ms (%.2f M ops/s)\n\n", ups, N/1e6/(ups/1e3));

  printf("═══════════════════════════════════════════════════════\n");
  printf("  RESULTS (%d GPUs, %zu elements)\n", ngpus, N);
  printf("═══════════════════════════════════════════════════════\n");
  printf("  Insert:           %8.1f ms  (%6.2f M ops/s)\n", ins, N/1e6/(ins/1e3));
  printf("  Find:             %8.1f ms  (%6.2f M ops/s)\n", fnd, N/1e6/(fnd/1e3));
  printf("  Contains:         %8.1f ms  (%6.2f M ops/s)\n", con, N/1e6/(con/1e3));
  printf("  Insert-or-assign: %8.1f ms  (%6.2f M ops/s)\n", ups, N/1e6/(ups/1e3));
  printf("═══════════════════════════════════════════════════════\n");

  CK(cudaFree(d_pairs)); CK(cudaFree(d_keys)); CK(cudaFree(d_vals));
  CK(cudaFree(d_found)); CK(cudaFree(d_err)); CK(cudaFree(d_cnt));
  printf("\nDone.\n"); return 0;
}
