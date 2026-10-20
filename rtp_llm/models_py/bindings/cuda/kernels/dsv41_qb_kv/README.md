# Native V4.1 merged-QB / SWA producer

This backend is opt-in and qualified on runtime SM103 with 148 SMs. The
existing DeepGEMM/Triton paths remain the fallback. It does not replace FlashMLA,
Global/Index-K compression, selection, or output projection.

## Source and build

The Q instruction wrappers and cluster MMA pipeline derive from
`foundation_models/cuda_extension`, revision
`94fd20eb9179301e8cfb14a48ce0070cc447de9f`, `dsv4_mega/wq_b` CSA implementation.
The persistent scheduling, non-swap pipeline, operand cache hint, side-worker
fusion and KV implementation are adaptations developed for this integration.
CUTLASS/CuTe headers remain external at the repository's pinned
`dc4817921edda44a549197ff3a9dcf5df0636e7b`; none are copied here.

Build only an SM103a cubin with CUDA13/x86. That old CUTLASS predates SM103:
`dsv41_qb_kv.cu` enables the SM90 TMA and TCGEN instruction guards only when
`__CUDA_ARCH__ == 1030 && __CUDA_ARCH_FEAT_SM103_ALL`. This prevents the old
headers' unsupported-architecture `brkpt` fallbacks without claiming SM100
support on other architectures. Other platforms expose an unavailable stub.

The native interface takes explicit output tensors, tile variant, cluster count
and incoming PDL policy; it has no process-global launch state and no runtime
JIT loader. Torch's current device/stream is used. C++ validates tensors before
launch, and Python performs metadata-only admission before allocating outputs.

## Execution contract

- E4M3 `[M,1280]` activation and `[36864,1280]` merged weight.
- Packed int32 UE8M0 scales `[rows,10]`, MN-major/TMA aligned.
- BF16 combined `[M,36864]` output; Main Q and Index Q are shared-storage views.
- Optional strided BF16 raw KV `[M,512]`, weighted RMSNorm, interleaved RoPE64,
  materialized BF16 KV and native MXFP8 cache `[pages,entries,528]`.
- Page padding, int32/int64 position/slot metadata and negative slots are
  supported. Positive out-of-range slots retain the existing debug trap policy.
  Write slots must be unique, as in the unfused producer.
- Main Q stays unnormalized/unrotated for the V4.1 Mega consumer. KV reduction,
  FMA operand order and BF16 boundary follow the target Triton implementation.
- Incoming Qnorm→QB PDL waits before any payload read. There is no outgoing
  producer→FlashMLA early trigger.

Initial dispatch (M is B×S token rows):

| M | Kernel geometry | Cluster count | KV workers per CTA |
| --- | --- | --- | --- |
| 128 | swap BM128 | 72 | 1 warp |
| 192 | swap BM192 | 72 | 1 warp |
| 256 | non-swap BM256/BN256, one accumulator stage | 72 | 4 warps |
| 512 | non-swap BM256/BN256, one accumulator stage | 72 | 4 warps |

M384 and all unqualified layouts keep DeepGEMM. The BN224/double-accumulator and
swap BM256 variants are retained as explicit native tuning controls, not enabled
by the Python dispatch. The non-swap fused launch retains the measured 512-thread
geometry; the unused tail warps do not participate in the GEMM barriers.

## Rollout and validation

`DSV41_CUSTOM_QB=1` enables the pure merged-QB control.
`DSV41_FUSED_QB_KV=1` enables QB + SWA, **2 kernels → 1**.
Both default to `0`. Existing FULL multistream and Qnorm PDL policy are preserved.
The decode wiring excludes Reindex so its default-on front projection and
QB/query-preparation fusion retain their own interfaces and prepared outputs.

Full current-layer TopK depends on this GEMM's Index Q, so its conversion cannot
join this launch. `DSV41_FUSED_TOPK_SLOTS=1` separately combines finite publication
and slot conversion. REUSE ready-index **3→1 is not implemented here**.

Run the new `test_v41_qb_kv_x86` and `test_v41_topk_slots_x86` Bazel targets on a
matching RTP wheel/build, plus the existing merged-QB/full-decode suites. The
portable performance runner is `rtp_llm/test/perf_test/dsv41_qb_kv.py`; it requires
a clean official DeepGEMM testing checkout at
`559d79fb6994a58b8a15b4b93bf13ccc16edf247` and the branch's locked DeepGEMM wheel.
It saves per-replay CUPTI traces and exact production-kernel envelopes after
8 GB flushes. Its isolated producer regions exclude attention and full-layer
competition; they are not a full-layer speedup claim.
