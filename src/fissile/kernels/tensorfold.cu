// Explicit-target SM12x wrapper; upstream kernels remain unchanged.
// TensorFold revision 8c2037c9aac16c2cd1353781dca627a0510b5a85, Apache-2.0.
// The build manifest records upstream sources, LICENSE and NOTICE.
#include <cuda_runtime.h>
#include <algorithm>
#include "prompt16.cu"

#ifndef FISSILE_CUDA_MINOR
#define FISSILE_CUDA_MINOR 1
#endif
static_assert(FISSILE_CUDA_MINOR == 0 || FISSILE_CUDA_MINOR == 1);

namespace {
thread_local const char* last_error = "success";
int check(cudaError_t err) {
    if (err != cudaSuccess) last_error = cudaGetErrorString(err);
    return static_cast<int>(err);
}
constexpr int classic_shared = tf_prompt16::Tile<tf_prompt16::FP4,128,128,2,2,2>::SMEM;
constexpr int lean_shared = tf_prompt16::Tile<tf_prompt16::FP4,128,256,1,8,3>::SMEM + 48;
}
extern "C" const char* fissile_tf_error() { return last_error; }
extern "C" int fissile_tf_initialize(int* sms, int* classic_smem, int* lean_smem) {
    int device = 0; cudaDeviceProp p;
    if (int e=check(cudaGetDevice(&device))) return e;
    if (int e=check(cudaGetDeviceProperties(&p,device))) return e;
    if (p.major != 12 || p.minor != FISSILE_CUDA_MINOR) { last_error="Device does not match compiled SM12x target"; return -1; }
    if (int e=check(cudaFuncSetAttribute(tf_prompt16::prompt_kernel<tf_prompt16::FP4,128,128,2,2,2,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,classic_shared))) return e;
    if (int e=check(cudaFuncSetAttribute(tf_prompt16::prompt_lean<tf_prompt16::FP4,128,256,1,8,3,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,lean_shared))) return e;
    *sms=p.multiProcessorCount; *classic_smem=classic_shared; *lean_smem=lean_shared;
    return 0;
}
extern "C" int fissile_tf_run(int tactic, const void* x, const void* words,
        const void* scales, float global, void* out, int m, int n, int k, int sms, void* stream) {
    if ((tactic!=0 && tactic!=1) || m<1 || n<1 || k<1 || n%128 || k%64 || sms<1) {
        last_error="Invalid aligned W4A16 matrix"; return -2;
    }
    const int rows_t=(m+127)/128;
    const int band=std::max(1,std::min(rows_t,(12<<20)/(128*k*2)));
    auto s=reinterpret_cast<cudaStream_t>(stream);
    auto a=static_cast<const __nv_bfloat16*>(x);
    auto w=static_cast<const unsigned char*>(words);
    auto b=static_cast<const uint8_t*>(scales);
    if (tactic==0) {
        int blocks=rows_t*((n+127)/128);
        tf_prompt16::prompt_kernel<tf_prompt16::FP4,128,128,2,2,2,false><<<blocks,128,classic_shared,s>>>(a,w,b,global,out,m,n,k,n,k,band);
    } else {
        int blocks=std::min(sms,rows_t*((n+255)/256));
        tf_prompt16::prompt_lean<tf_prompt16::FP4,128,256,1,8,3,false><<<blocks,256,lean_shared,s>>>(a,w,b,global,out,m,n,k,n,k,band);
    }
    return check(cudaGetLastError());
}
