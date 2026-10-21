# V4.1 DSpark MTP preparation fusion

The proposal input preparation and non-CP commit row mapper launch several small integer kernels. This change combines proposal casts/noise/anchor/positions/mask in one Triton kernel and combines the commit upper-bound search, gathers, validity and output casts in another. Existing metadata and SWA-slot consumers remain unchanged. No GEMM, draft Transformer, head, or sampler is replaced.

The local measurement source base is `438fbb2a0b8aa156b29184ee03daa774c64e7c25`. On the target device, PyTorch reports capability `(10,3)` and 148 SM. The following results use gamma=5, int32 ids; B denotes request count. Values are microseconds, forward / reverse measurement order:

| B | proposal原版 A/B μs | proposal融合 A/B μs | commit原版 A/B μs | commit融合 A/B μs |
| --- | --- | --- | --- | --- |
| 128 | 10.624 / 10.592 | 4.480 / 4.480 | 61.600 / 61.665 | 7.344 / 7.360 |
| 192 | 12.976 / 12.896 | 4.480 / 4.480 | 58.272 / 58.192 | 7.392 / 7.472 |
| 256 | 12.800 / 13.040 | 4.704 / 4.704 | 60.304 / 60.321 | 8.512 / 8.544 |
| 384 | 12.976 / 13.023 | 4.960 / 5.008 | 58.577 / 58.608 | 8.160 / 8.160 |
| 512 | 14.336 / 14.304 | 5.856 / 5.808 | 57.664 / 57.633 | 8.625 / 8.640 |

The proposal envelope includes the existing metadata consumer: 7 kernels become 2 (8 become 2 with int64 ids). Commit includes the actual ends-to-int32 cast and SWA-slot consumer: 27 become 3. The mapper itself changes from 25 kernels to 1, and proposal preparation from 6/7 to 1. The commit measurement is an explicitly selected integer dependency subgraph; actual feature projection/normalization between mapping and KV injection is excluded. These savings must not be added together or reported as full-round MTP speedup.

Validation covers B128/192/256/384/512, gamma1/5/8, plus int64 ids at gamma5: 35 configurations, 140 paired variant runs, 4200 samples. Every sample is one warmed CUDA Graph replay after an 8GB flush, measured with the exact CUPTI first-start to last-end envelope; four flush kernels are excluded. There are 3 warmups, one capture, and 20 replay warmups. The shared GPU is unlocked, task GPU jobs are serial, and the second round reverses order. No configuration regressed in both final scans.

Perfetto TraceProcessor independently checked all 140 raw traces, GPU kernel names/counts, durations, relative starts, and all replay envelopes. The original Kineto complete events can produce overlap-recovery warnings; GPU values are checked against raw CUPTI regardless. The comparison view splits overlapping activities into display lanes without changing timestamps. Activity boundary overlap does not prove useful internal compute overlap.

210 candidate shape/phase cases (1200 output tensor comparisons) pass bitwise against unchanged baseline computation, including final physical slots, duplicate starts, all-zero lengths, padding, and changed same-address buffers. Outputs are poisoned before replay and input phase hashes match between rounds. Numeric checks import official DeepGEMM testing `559d79fb6994a58b8a15b4b93bf13ccc16edf247` after provenance checks.

15 kernel/contract tests pass without failures or skips. The installed older C++ ABI prevents direct import of the full framework tests (`HiddenStateCaptureDtype` missing). The no-Bazel runner compiles original mixin class/function bodies unchanged and uses the real cache normalizer, PyModelOutputs and GPU helpers; only module-level configuration imports are isolated. The original unit-test draft hooks are test doubles and provide no full-MTP performance evidence. Full model loading, serving, TP/CP, and an entire speculative round remain unvalidated here.

Both helpers allocate only forward-local outputs and never memoize device addresses. Sorted starts are the original searchsorted precondition; duplicate starts use right=True. The mapper admits contiguous CUDA int64 inputs and otherwise keeps eager behavior. Proposal inputs admit contiguous CUDA int32/int64 with the original int32-before-position cast order. CPU and unsupported cases retain eager code, whose AST is checked against the pre-change snapshot. Empty batches still reach the collective-layer hook. The V4.1 CP-specific mapper override is unchanged.

Set `DSV4_FUSED_DSPARK_QUERY_INPUTS=0` or `DSV4_FUSED_DSPARK_COMMIT_ROWS=0` to disable the respective helper. They are enabled by default for admitted inputs.

DeepInfer reference: [DFlash metadata](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/input_metadata/dflash.py), [fused preparation](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/cuda/get_kv_index.py). Its first-token/accepted-offset/ring contract differs from RTP's anchor/prefix/paged-table contract; only the fusion boundary is borrowed, not its address formula. RTP's already-fused draft attention metadata is not counted as new work.

[Audited summary and source/trace locks](dsv41_mtp_preparation_fusion.json),
[interactive attention/MTP HTML](dsv41_attention_mtp_fusion/attention_mtp_fusion.html),
[Perfetto comparison trace](dsv41_attention_mtp_fusion/attention_mtp_overview.perfetto.json),
and [GPU reproducer with archived baseline](dsv41_mtp_reproducer/README.md) are
included in the repository. The full raw trace archive and original Perfetto SQL
audits remain in the shared workspace at
`experiments/dsv41_mtp_fusion/integration_v14`. Original remote container:
`zzx-dsv41-fusion-gpu1-20261020`; set
`RTP_MTP_SOURCE_ROOT=/work/integration_v14/rtp-source`, run
`benchmark.py --out NEW_DIR --modes baseline fused`, then repeat with `--reverse`.
Use fresh output directories.
