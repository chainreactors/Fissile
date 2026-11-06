# Third-party notices

Fissile includes and adapts the following upstream components.

| Component | Pinned source | Use | License |
|---|---|---|---|
| TensorFold | [ashhart/TensorFold @ 8c2037c9aac16c2cd1353781dca627a0510b5a85](https://github.com/ashhart/TensorFold/tree/8c2037c9aac16c2cd1353781dca627a0510b5a85) | Unmodified prompt16.cu and qmm_frag.cuh; C ABI wrapper | Apache-2.0; upstream LICENSE, NOTICE and THIRD_PARTY_NOTICES retained |
| FlashInfer/CUTLASS dense GEMM | [flashinfer-ai/flashinfer @ 835ad80a3a189b086c9b700c417ee0085ea3f9b8](https://github.com/flashinfer-ai/flashinfer/tree/835ad80a3a189b086c9b700c417ee0085ea3f9b8) | Paired gate/up output, fixed SiLU lookup, full-grid scheduling, package import changes | NVIDIA BSD-3-Clause header retained |
| FlashInfer helpers | Same revision | Extracted scale-layout helpers without arithmetic changes | Apache-2.0, copyright retained |
| NVIDIA CUTLASS example 79a | CUTLASS collective-builder API; built against a user-supplied CUTLASS 4.5 tree | `kernels/fp4_gemm.cu`: one pinned SM121 block-scaled NVFP4 GEMM configuration, row-stride output, C ABI (`gb10-mixed-v2` only) | NVIDIA BSD-3-Clause header retained |
| NVIDIA Model Optimizer | [NVIDIA/Model-Optimizer @ 1c40150d459e4817e43c9fe3db75f3b6e3a24417](https://github.com/NVIDIA/Model-Optimizer/tree/1c40150d459e4817e43c9fe3db75f3b6e3a24417) | Local-Hessian FP8 scale sweep (`quantization/hessian.py`) and shared FP4 conversion helpers; direct packed output, tensor-global scale | Apache-2.0, NVIDIA headers retained |

[PyTorch](https://github.com/pytorch/pytorch),
[Triton](https://github.com/triton-lang/triton),
[SGLang](https://github.com/sgl-project/sglang),
[Transformers](https://github.com/huggingface/transformers),
[CUTLASS CuTe DSL](https://github.com/NVIDIA/cutlass) and
[CUDA Python bindings](https://github.com/NVIDIA/cuda-python) are external
dependencies, not bundled environments. CUDA Toolkit is user-supplied.
Their own distribution terms apply; no NVIDIA endorsement is implied.

Algorithm references: [GPTQ](https://arxiv.org/abs/2210.17323),
[MR-GPTQ](https://arxiv.org/abs/2509.23202) and, for the optional adaptive
activation scale, [Four Over Six](https://arxiv.org/abs/2512.02010v5).

Models, adapters, calibration examples and evaluation datasets are not included.
