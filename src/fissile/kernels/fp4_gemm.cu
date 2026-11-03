/*
 * Pinned SM121 NVFP4 W4A4 GEMM: one CUTLASS collective-builder configuration.
 * Derived from NVIDIA CUTLASS example 79a and its collective-builder API.
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Operands match fissile.kernels.nvfp4.matmul: row-major E2M1 A [m, k/2],
 * E2M1 B [n, k/2] (column-major K x N), SWIZZLE_32_4_4 E4M3 block-16 scales,
 * device FP32 alpha = global_a * global_b, BF16 D. Every output element
 * accumulates its full K range sequentially in one CTA (no split/stream-K).
 * D may be a column slice of a wider row-major buffer (leading dimension ldd).
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <exception>
#include "cutlass/cutlass.h"
#include "cutlass/version.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {
thread_local char last_error[256] = "success";
constexpr int kSharedMemoryLimit = 101376;  // SM121 opt-in shared memory per block

using Arch = cutlass::arch::Sm120;
using Tile = Shape<_128, _128, _256>;
using Cluster = Shape<_1, _1, _1>;
using E = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using D = cutlass::bfloat16_t;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Fusion = cutlass::epilogue::fusion::LinearCombination<D, float, void, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, cutlass::arch::OpClassTensorOp, Tile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    void, Row, 8, D, Row, 8,
    cutlass::epilogue::TmaWarpSpecialized, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, cutlass::arch::OpClassBlockScaledTensorOp, E, Row, 32, E, Col, 32, float,
    Tile, Cluster, cutlass::gemm::collective::StageCount<2>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GK = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, Main, Epi, cutlass::gemm::StaticPersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;

int sm_count = 0;
int device_id = -1;

int fail(int code, const char* message) {
  snprintf(last_error, sizeof(last_error), "%s", message);
  return code;
}

typename Gemm::Arguments arguments(void* out, int64_t ldd, const void* a, const void* b,
                                   const void* sa, const void* sb, const float* alpha,
                                   int m, int n, int k) {
  using Config = typename Main::Sm1xxBlkScaledConfig;
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = make_shape(m, n, k, 1);
  args.mainloop.ptr_A = static_cast<const cutlass::float_e2m1_t*>(a);
  args.mainloop.ptr_B = static_cast<const cutlass::float_e2m1_t*>(b);
  args.mainloop.ptr_SFA = static_cast<const cutlass::float_ue4m3_t*>(sa);
  args.mainloop.ptr_SFB = static_cast<const cutlass::float_ue4m3_t*>(sb);
  args.mainloop.dA = cutlass::make_cute_packed_stride(typename GK::StrideA{}, {m, k, 1});
  args.mainloop.dB = cutlass::make_cute_packed_stride(typename GK::StrideB{}, {n, k, 1});
  args.mainloop.layout_SFA = Config::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = Config::tile_atom_to_shape_SFB(args.problem_shape);
  args.epilogue.ptr_C = nullptr;
  args.epilogue.ptr_D = static_cast<D*>(out);
  // Row-major D with leading dimension ldd: packed stride of an [m, ldd] matrix.
  args.epilogue.dD = cutlass::make_cute_packed_stride(
      typename GK::StrideD{}, {m, static_cast<int>(ldd), 1});
  args.epilogue.dC = args.epilogue.dD;
  args.epilogue.thread.alpha_ptr = alpha;
  args.hw_info.device_id = device_id;
  args.hw_info.sm_count = sm_count;
  args.hw_info.cluster_shape = dim3(1, 1, 1);
  args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>)
    args.scheduler.max_swizzle_size = 1;
  return args;
}
}  // namespace

extern "C" const char* fissile_fp4_gemm_error() { return last_error; }

extern "C" int fissile_fp4_gemm_version(int* cutlass_version) {
  *cutlass_version = CUTLASS_MAJOR * 10000 + CUTLASS_MINOR * 100 + CUTLASS_PATCH;
  return 1;
}

extern "C" int fissile_fp4_gemm_initialize(int* sms, int* shared_memory) {
  cudaDeviceProp p;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&p, device) != cudaSuccess)
    return fail(-1, "Cannot query the current CUDA device");
  if (p.major != 12 || p.minor != 1) return fail(-1, "Device is not the compiled SM121 target");
  if (sizeof(typename GK::SharedStorage) > kSharedMemoryLimit)
    return fail(-2, "Kernel shared memory exceeds the SM121 limit");
  device_id = device;
  sm_count = p.multiProcessorCount;
  *sms = sm_count;
  *shared_memory = static_cast<int>(sizeof(typename GK::SharedStorage));
  return 0;
}

extern "C" int fissile_fp4_gemm_run(void* out, int64_t ldd, const void* a, const void* b,
                                    const void* sa, const void* sb, const float* alpha,
                                    int m, int n, int k, void* stream) {
  last_error[0] = 0;
  if (device_id < 0) return fail(-1, "fissile_fp4_gemm_initialize was not called");
  if (m < 1 || n < 1 || k < 1 || m % 16 || n % 128 || k % 256 || ldd < n || ldd % 8)
    return fail(-3, "Unsupported NVFP4 GEMM geometry");
  try {
    auto args = arguments(out, ldd, a, b, sa, sb, alpha, m, n, k);
    Gemm gemm;
    if (gemm.get_workspace_size(args) != 0) return fail(-4, "Unexpected workspace requirement");
    auto status = gemm.can_implement(args);
    if (status == cutlass::Status::kSuccess)
      status = gemm.initialize(args, nullptr, static_cast<cudaStream_t>(stream));
    if (status == cutlass::Status::kSuccess)
      status = gemm.run(static_cast<cudaStream_t>(stream));
    if (status != cutlass::Status::kSuccess) return fail(1, cutlassGetStatusString(status));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return fail(static_cast<int>(error), cudaGetErrorString(error));
  } catch (const std::exception& e) {
    return fail(-5, e.what());
  }
  return 0;
}
