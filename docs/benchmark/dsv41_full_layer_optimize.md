# DSv4.1 FULL Attention: Multistream and PDL Handoff

Checkpoint: 2026-10-20, branch `full_layer_optimize` in
[guluguluhhhh/rtp-llm](https://github.com/guluguluhhhh/rtp-llm/tree/full_layer_optimize).
The eligible FULL decode path enables multistream and minimal QB PDL by default.
This does not constitute completed whole-model accuracy/performance validation.

Submission scope: retain the existing three-stream FULL schedule and only the
Q-norm -> merged QB PDL edge, both default-on within their eligibility gates.
Broader producer/consumer PDL,
custom Index K GEMM, Index Q fork/warp tuning, and Reindex/Reuse scheduling are
not included. The experimental results and follow-up plans below are historical
context, not additional code paths or acceptance results for the rebased branch.

## Scope and Current State

Only the DSv4.1 FP8 **FULL1/FULL2 decode attention** path is being optimized.
Keep Reuse/Reindex, prefill, MoE, Engram and third-party implementations unchanged.
In particular, do not modify DeepGEMM to continue this experiment.

| Change | Included work |
| --- | --- |
| Global producer | Remove the row-count cap on the existing fused Global producer; retain layout/state eligibility checks |
| Indexer metadata | Fuse/share compatible decode metadata within a forward; retain the uncached fallback when RoPE reuse is disabled |
| FULL multistream | Add default-on, three-stream FULL decode scheduling and correctness tests |
| Minimal PDL | Default-on Q-norm producer PDL trigger, gated on DeepGEMM's existing policy, plus tests and the portable experiment runner |

The current PDL code covers **one edge: Q norm/quant -> merged QB**. It does not
implement a pdlrt-style whole-layer PDL chain. The existing multistream design
remains intact. A separate scheduling scan explored enqueue-order variations;
those scan hooks are not part of the model implementation in this branch.

The runner uses production projection dimensions and paged-cache layouts with
random weights and synthetic KV. It needs **one available SM10x GPU**, not a
checkpoint or an eight-GPU server. `B=64, S=6, M=384` matches the target-verify
shape for MTP gamma=5, but is **not a complete MTP step**, request TPS, real-prompt
accuracy test, or whole-model benchmark.

## Implementation and Switches

| Setting | Meaning |
| --- | --- |
| `DSV41_FULL_DECODE_MULTISTREAM=1` | Enable eligible FULL fork/join; default is `1`, set `0` to disable |
| `DSV41_FULL_DECODE_QB_PDL=1` | Allow the Q-norm producer trigger in the eligible FULL multistream path; default is `1`, set `0` to disable |
| `deep_gemm.get_pdl()` | Must already be true for that producer trigger to be enabled |
| `DSV41_FUSE_INDEXER_METADATA=1` | Fused/shared metadata; keep enabled for these comparisons |
| `DSV41_REUSE_DECODE_ROPE=1` | Required for shared indexer metadata; `0` uses the uncached preparation path |
| `DSV41_OVERLAP_GLOBAL_QKV=0` | Disable the older, separate overlap experiment to isolate this path |

Setting the QB environment variable alone is **not sufficient**. Production
`full_overlap.py` only reads DeepGEMM's process-global policy; it never changes
it. The standalone runner sets/restores the policy around warmup/capture in its
own process. Do not toggle it underneath a running inference service. Changing
an environment variable or library policy does not rewrite an already captured
graph; capture a new graph for each policy.

To restore the serial FULL path without the new producer trigger, set both
`DSV41_FULL_DECODE_MULTISTREAM=0` and `DSV41_FULL_DECODE_QB_PDL=0` before startup.
Keep `DSV41_OVERLAP_GLOBAL_QKV=0` to disable the older overlap experiment too.
Disabling only QB PDL preserves FULL multistream. These switches do not disable
DeepGEMM's own PDL policy for other kernels.

The FULL gate checks TP1, SM10x/paged native-producer support, matching FULL
KV/index ownership, Index H32/D128/TopK512, input/cache/state layout and other
existing eligibility conditions. Unsupported or cold-capture cases fall back
before forking. Eager warmup initializes side streams and per-shape capture
eligibility. Do not remove these gates just to obtain a timing result.

### Current DAG

`Global A/B` refer to the existing fused Global producer kernels. The streams
are dependencies inside the engine's parent graph, not nested per-layer graphs.

```text
Input ready
  |
  +-- caller: input quant -> QA+WKV -> Q norm/quant -> merged QB
  |                                      |              |
  |                                      +-- PDL ------>+  (optional)
  |                                                     v
  |                                               QKV/SWA post
  |                                                     |
  |                                               wait auxiliary
  |                                                     v
  |                                                Index Q post
  |                                                     |
  +-- global: V[/G] -> Global A -> Index K GEMM -> Global B
  |                                                     |
  +-- auxiliary: Indexer weights -> metadata              |
                                                        v
                         caller: wait global -> Score -> compact logits
                                                            |
                                  +-------------------------+------------------+
                                  |                                            |
                             token TopK                               FULL1 candidate
                             finite/slots                             pool -> block TopK
                             MLA -> WO_A -> quant -> WO_B             -> flags
                                  |                                            |
                                  +---------------- layer exit join ------------+
```

- Index Q post waits for Indexer weights, but can overlap Global production.
- Score needs Index Q, weights, metadata and the newly stored Index K cache.
- FULL1 candidate selection and current-layer token TopK both read compact
  logits. Candidate completion is not a prerequisite for current-layer token
  TopK/MLA. The candidate output is published for later layers; the current
  implementation joins it at layer exit, preserving the old interface.
- FULL2 retains Global A's old-ring read before Global B overwrites state.
  This is a real read-before-write dependency, not an optional scheduling edge.
- Side-stream inputs are retained until join; side-allocated outputs are
  registered with the caller stream. Preserve these lifetimes when moving work.
- Python stream count does not imply the same number of physical execution lanes
  in a replayed CUDA Graph. Validate graph edges and actual device overlap.

### What the PDL Edge Does

The Triton Q-norm kernel emits `griddepcontrol.launch_dependents` at entry when
enabled. QA -> Q norm still uses ordinary completion ordering. The unchanged
DeepGEMM QB consumer must launch with programmatic serialization and wait before
reading Q/scales. Early consumer launch permits independent initialization;
it does not mean Q is already available, nor does it promise useful weight-load
overlap in every GEMM dispatch. The first experiment checked the compiled
producer trigger and the captured programmatic edge.

## Start on Another Machine

### Prerequisites

- Check out this fork's `full_layer_optimize` branch and record the exact SHA.
  Do not silently update dependencies or rebase while collecting an A/B series.
- Use a compatible RTP Python/native runtime. The runner needs
  `libth_transformer_config.so` and `librtp_compute_ops.so` plus their dependent
  libraries; it does not build or install them. Follow the repository's build
  workflow if a compatible build is not available.
- First measurements used B300/SM103, Python 3.10, PyTorch `2.11.0+cu130`,
  Triton `3.6.0`, CUDA toolkit/JIT 13.2, DeepGEMM
  `2.8.2+30e85d9.cu132`, and a FlashMLA wheel exposing Mega Attention.
  This is an observed environment, not proof that arbitrary mixed wheels work.
- CUDA Python driver bindings must expose graph edge data/ports. The runner
  deliberately fails if it cannot audit the programmatic dependency.
- All previously reported timings had `wo_a_quant_ready=false`: installed
  DeepGEMM headers did not satisfy the custom `_v41_wo_a_quant` dependency
  checks. Both arms ran WO_A GEMM -> separate FP8 quant -> WO_B. Do not bypass
  those version/hash checks. A destination with the fused epilogue needs a new
  matched A/B and separate reporting.
- Confirm GPU ownership as well as utilization. The lock is cooperative and
  cannot exclude unrelated programs that ignore it. Do not kill other jobs.

From the public repository root, use your prepared Python environment and native
build. Replace these example paths and the GPU index:

```bash
export RTP_PYTHON=/path/to/environment/bin/python
export RTP_NATIVE_LIB_DIR=/path/to/compatible/build/rtp_llm/libs
export CUDA_HOME=/path/to/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$RTP_NATIVE_LIB_DIR:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RTP_EXPERIMENT_ROOT=/path/to/local/results/full_layer_optimize
export RTP_TEST_GPU=0
```

The script creates a runtime overlay under the output directory: Python sources
come from this checkout and `rtp_llm/libs` points at the selected native build.
It verifies the imported FULL source path and records the two native-library
SHA256 hashes. No checkout symlinks or installed packages are modified. Other
runtime dependencies must already be on the environment's library search path.
Keep results outside the source tree; use a new output directory for every run.

### Four Required Controls

| Runner variant | FULL multistream | DeepGEMM PDL | Q-norm early trigger |
| --- | --- | --- | --- |
| `serial` | Off | Off | Off |
| `multistream` | On | Off | Off |
| `dg_pdl` | On | On | Off |
| `qb_pdl` | On | On | On |

`serial` vs `multistream` isolates scheduling. `dg_pdl` vs `qb_pdl` isolates the
new producer trigger. `multistream` vs `dg_pdl` measures the library-global policy
change, which can affect other DeepGEMM calls too. `serial` vs `qb_pdl` measures
the combined effect. Do not describe the latter as the effect of just one edge.

### Commands

Run correctness first. The selected suite had 32 tests in the original PDL
experiment; any failure **or skip** makes this runner reject the run.

```bash
"$RTP_PYTHON" rtp_llm/test/perf_test/dsv41_full_decode_pdl.py \
  --gpu "$RTP_TEST_GPU" --native-lib-dir "$RTP_NATIVE_LIB_DIR" \
  --output "$RTP_EXPERIMENT_ROOT/tests-01" --scope tests
```

Then isolate Q norm + merged QB, including non-tile-aligned row counts:

```bash
"$RTP_PYTHON" rtp_llm/test/perf_test/dsv41_full_decode_pdl.py \
  --gpu "$RTP_TEST_GPU" --native-lib-dir "$RTP_NATIVE_LIB_DIR" \
  --output "$RTP_EXPERIMENT_ROOT/chain-01" --scope chain \
  --variants multistream dg_pdl qb_pdl \
  --rows 1 7 64 97 384 769 1024 --captures 3 --samples 100
```

Chain mode calls Q norm/QB directly; its `multistream` label is simply the
library-PDL-off chain control, not a three-stream layer.

Finally run both FULL types at B64/S6 and 32K/128K, with all four policies:

```bash
"$RTP_PYTHON" rtp_llm/test/perf_test/dsv41_full_decode_pdl.py \
  --gpu "$RTP_TEST_GPU" --native-lib-dir "$RTP_NATIVE_LIB_DIR" \
  --output "$RTP_EXPERIMENT_ROOT/full-01" --scope full \
  --batch 64 --span 6 --contexts 32768 131072 --ratios 2 1 \
  --variants serial multistream dg_pdl qb_pdl --captures 3 --samples 100
```

Repeat in a fresh process/output directory (`full-02`, optionally another seed).
Use `--contexts 131072` to complete the previously unfinished 128K experiment
first. Add `--profile` in a separate run to export raw timelines after timing;
profiling never contributes performance samples. `--skip-unit-tests` is only for
iteration after a successful suite on the same source/dependency combination.

The default admission limit is 32 MiB and zero GPU utilization, checked before
and after acquiring `/tmp/rtp_llm/smoke/test/gpu_status/<physical-index>`.
CUDA selects the checked GPU by UUID. Use `--lock-dir` only to match the actual
host's cooperative lock convention. Raise `--idle-memory-mib` only after
confirming the resident allocation's ownership, not to evade a busy GPU.

### Measurement and Artifacts

The runner does five eager warmups before each capture, 30 graph warmups, then
100 randomized-order samples by default. CUDA events are **inside the graph**,
around the entire FULL wrapper including the final side-stream join. A 256 MiB
flush buffer is touched outside the measured region before each sample. This
removes host launch gaps from the measured interval; real device stalls and
join waits must not be subtracted. Record clocks/power/other workloads separately
when comparing machines or runs.

Before timing, FULL graphs replay with three changed inputs, query positions,
SWA slots and physical page mappings per capture. All variants start from the
same cache/state snapshot and compare against eager serial, library-PDL-off
execution. Outputs, token indices, FULL1 candidates/flags and cache/state must
be bitwise equal. Chain mode tests ten changed inputs. These are synchronization
tests: do not loosen tolerances to hide a race.

| Artifact | Purpose |
| --- | --- |
| `source.json`, `source.patch` | Commit, worktree state and tracked patch; untracked files are not archived |
| `arguments.json`, `settings.json`, `versions.json` | Shapes/policies, selected settings, source paths, packages/native hashes |
| `gpu_before.json`, `gpu_locked.json` | Admission snapshots |
| `tests.json` | Test count, errors, failures and skips |
| `*_edges.json` | Captured nodes, grid/block/smem and edge types/ports; norm -> QB is explicitly audited |
| `results.json` | Every raw timing sample, per-capture statistics, replay count, WO_A path |
| `*_serial.json`, etc. with `--profile` | Original independent profiler traces, not performance inputs |
| `completed.json` | All requested work completed successfully |
| `failed.json` | Error after admission; early admission errors instead appear on stderr |

Save stdout/stderr too. No completion marker means the whole requested run was
not completed, even if partial timing rows exist. The final GPU snapshot is
before process exit and can still include its CUDA context. Confirm release
after process exit. No broad process/cache cleanup is part of this runner.

For reporting, take the median of per-capture medians, also show each capture
and its spread. Never choose only the fastest capture. Profile counts/names are
checked across variants; a dispatch change must be understood before accepting
the comparison. Kernel duration sums double-count overlap and can include PDL
waits, so they are not layer wall time.

**Runner validation boundary:** this portable script packages the previously
used local experiment and adds explicit four-policy controls and GPU admission.
After the initial handoff, it completed the four controls at 32K/128K FULL1/2,
three captures per case, and 32/32 tests with no skips on an available SM103 GPU.
This validated that earlier revision in that runtime, not arbitrary destination
wheel/native-library combinations.
The earlier results below retain their original predecessor-script provenance.
Six CPU-only runner checks passed (CLI/admission, locked UUID selection, policy
restoration, statistics and output preservation). An earlier GPU-hidden run
passed 11 of 12 selected gate/lifetime tests. The failing
`FullOverlapCPUTest.test_cold_capture_and_failed_producer` fixture was introduced
by this branch: its mock streams lacked concrete `device.index` values and
caused real CUDA stream allocation. The fixture now sets the mock devices;
neither production stream validation nor the test assertions are weakened.

The 2026-10-20 review also inspected the source shipped in the pinned
DeepGEMM `2.8.2+ad62a90.cu132` wheel. In
`include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`,
`cudaGridDependencySynchronize()` precedes the A/B and scale TMA loads.
The packed-int32 scale path validates and returns its inputs without launching
a transform kernel. In `scheduler/sm100_paged_mqa_logits.cuh`, the dense
schedule, lengths, request IDs and block table are read-only; traversal cursors
are kernel-local. This is a source audit, not GPU acceptance of that wheel.
Regression coverage now includes repeated consumption by distinct KV owners,
unchanged metadata, changing-input graph replays, and the
`DSV41_REUSE_DECODE_ROPE=0` / `DSV41_FUSE_INDEXER_METADATA=1` fallback.

Review-fix validation on 2026-10-20 initially passed 24 selected CPU-only tests
and the runner's 36-test GPU-enabled suite with a temporary 192-token producer
gate. That gate restoration was subsequently withdrawn; the unlimited producer
remains part of this branch. The final uncapped tree passed all 23 selected
CPU-only tests with no skips and without initializing CUDA. A fresh GPU run of
that final tree was not performed because no confirmed available healthy GPU
remained. The GPU suite used the pinned
DeepGEMM `2.8.2+ad62a90.cu132` wheel from an isolated Python path, PyTorch
`2.11.0+cu130`, Triton `3.6.0`, CUDA toolkit 13.2, and an existing RTP native
build. It included the minimal QB-PDL graph, FULL multistream, shared metadata
read-only checks and selector fallback. This is scoped unit/graph validation,
not a fresh native build, whole-model accuracy result or performance rerun.

The subsequent rebase onto `feature/dsv41-dev-main-202161013` at `9ca919458e`
preserves upstream native megaMHC FP8 input reuse through both the serial and
FULL multistream paths, independently of the QB PDL switch. All 30 selected
CPU contract tests passed without skips or CUDA initialization. They include
mocked eager/capture forwarding and shared-input/PDL combinations; they do not
constitute a new native build, GPU graph validation or performance measurement.

## Existing Results, Not New Acceptance Results

Initial multistream A/B, B64/S6, medians of 100 unprofiled samples, microseconds:

| Context | Layer | Serial Graph | Multistream Graph |
| --- | --- | ---: | ---: |
| 32K | FULL2 | 266.240 | 245.728 |
| 32K | FULL1 | 298.112 | 271.360 |
| 128K | FULL2 | 408.576 | 386.240 |
| 128K | FULL1 | 502.784 | 467.968 |

That phase passed 30/30 regression tests. Its traces had 29 FULL2 and 30 FULL1
kernel launches, including split-K reductions and metadata. Logical DAG nodes
are not the same as launch counts. These numbers are not whole-model latency.

First PDL experiment, **both arms already had DeepGEMM PDL enabled**. Only the
Q-norm early trigger changed. Three captures, 100 samples each, median of capture
medians, microseconds:

| Context | Layer | Multistream + DG PDL | Plus Q-norm trigger | Reduction |
| --- | --- | ---: | ---: | ---: |
| 32K | FULL2 | 244.736 | 242.688 | 0.84% |
| 32K | FULL1 | 271.328 | 269.344 | 0.73% |

32/32 tests passed, no skips; changed-input graph outputs/state matched bitwise.
Programmatic graph edges and producer trigger were inspected. The M384 isolated
Q-norm/QB chain was 28.672 -> 28.672 us in the initial run: **no measured gain**.
The original library default was PDL off, so this table is not a clean comparison
against that default and does not establish a stable end-to-end improvement.
The 128K PDL run stopped after the initial capture/edge audit; there is no
completed 128K PDL result. Do not extrapolate the 32K percentage.

The separate enqueue-order scan found better points, especially FULL1/128K:

| Context | Layer | Serial | Current multistream | Best scanned policy |
| --- | --- | ---: | ---: | ---: |
| 32K | FULL2 | 267.22 | 246.78 | 245.74 |
| 32K | FULL1 | 298.02 | 271.93 | 265.20 |
| 128K | FULL2 | 407.58 | 388.37 | 386.05 |
| 128K | FULL1 | 500.86 | 470.27 | 445.82 |

These are a different A/B series, not additive savings. The scan included capture
outliers and has not been promoted into production scheduling. Its useful ideas:
record the Global fork at layer entry, enqueue QA before submitting Global work
(`qa_submit`); record the candidate dependency at compact-logit completion,
enqueue token TopK/finite first, then submit candidate work (`tokens_submit`).
Neither means waiting for QA/token TopK to complete. Enqueue order is a hint,
not a hardware priority guarantee. Reproduce before adopting a policy.

## Follow-up PDL Configuration Scan

The same-day follow-up completed the missing 128K controls and explored ten
multistream policies: library PDL off, library PDL without an explicit producer
trigger, and four trigger positions (entry, after loads, after RMS inverse,
after stores) with either all DG calls or only merged QB using PDL.
All used the original eight-warps reduction. These extra policies were local
function-boundary experiments, not changes to this branch's production defaults.

Two fresh confirmation processes on the same SM103 GPU each used five captures
and 100 randomized-order samples per capture. Below are medians of the ten
capture medians, with profiler runs and earlier runs on another GPU excluded.

| Context | Layer | Serial us | Multistream us | Existing entry PDL us | Best observed us | Best observed trigger |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 32K | FULL2 | 264.376 | 243.744 | 241.696 | 239.648 | After RMS inverse |
| 32K | FULL1 | 298.944 | 271.496 | 268.352 | 268.288 | After loads |
| 128K | FULL2 | 407.664 | 387.056 | 382.976 | 382.008 | After RMS inverse |
| 128K | FULL1 | 501.760 | 466.000 | 462.920 | 462.848 | After loads |

The best observed policies used global library PDL. They improve over
multistream by 0.68-1.68%, but exceed the existing entry-trigger experiment by
only 0.064-2.048 us. Trigger rankings were not identical in the earlier GPU's
scan. Capture outliers were retained, including FULL1/128K mid-trigger at
491.312 us and FULL1/32K entry-trigger at 288.880 us. This is not a universal
optimum or evidence of improved worst-case latency. Keep the existing entry
trigger as the reference; none of these local candidates were promoted.

Dynamic FULL output/indices/candidate/cache/state comparisons passed bitwise.
A separate seven-shape Q-norm/QB sweep (M=1/7/64/97/384/769/1024) passed exact
normalized-Q, FP8, scale and both projection-output checks for the eight-warps
policies. A four-warps experiment differed in one M1 BF16 norm element by
0.001953125; that difference reproduced with PDL off while quantized values and
QB outputs remained equal for that input. It was excluded from this
synchronization-only comparison rather than hidden by a relaxed tolerance.

Sixteen independent raw traces confirmed 29 FULL2 and 30 FULL1 launches per
replay. Only the deliberately replaced norm kernel's name needed normalization
for the multiset check. The norm/QB pair had zero overlap with PDL off and
approximately 1.27-4.13 us overlap with the entry trigger, depending on the case.
These profiler intervals are not latency savings. PTX and graph-edge checks
confirmed the trigger locations and programmatic consumer dependency.

One important constraint was verified in the installed DeepGEMM implementation:
descriptor prefetch, barrier initialization and TMEM allocation precede
`cudaGridDependencySynchronize()`, while actual A/B TMA loading follows it.
Changing our trigger cannot create pdlrt/GLM's independent weight-loading path.
With DeepGEMM unchanged, extending protocols in RTP-owned postprocessing is a
more useful next investigation than assuming trigger tuning alone will yield a
large gain. Index Q -> Score is a candidate, but its Global-stream join and
multiple input parents must be represented correctly first.

## Broader FULL Experiment: 2026-10-19

The follow-up tested the GLM-style multi-producer join into Score and the
pdlrt-style distinction between independent branches and dependent preparation.
These are **isolated wrapper prototypes**, not enabled production paths:

- Index Q -> Score and Global B -> Score early triggers, separately and together.
- Metadata -> DG scheduler trigger; QKV -> Index Q trigger/wait.
- Index Q positions and immutable RoPE loads before its dependency wait.
- Index Q launch resources: 4, 2 and 1 warp; entry versus post-load trigger.
- Ordinary QB -> {QKV, Index Q} fork. Index Q runs on the existing auxiliary
  stream after both QB and indexer weights/metadata. The caller joins before
  Score; QKV/SWA still completes before MLA. No DeepGEMM kernel was changed.

Blindly extending PDL was not a win. In a three-capture screen, the broad
four-producer policy measured 245.936/272.192/389.152/464.928 us for 32K
FULL2/FULL1 and 128K FULL2/FULL1, versus 241.568/268.288/385.024/466.864 us
for the existing QB PDL policy. The isolated Global B trigger had a small,
inconsistent effect. Do not enable all triggers by default.

The same-GPU confirmation below uses **two processes, five captures each,
100 unprofiled samples per capture**, B64/S6, both FULL ratios. Values are
medians of the ten capture medians; per-capture p90/min/max and samples were
retained. The uniform new candidate keeps existing QB PDL, forks QKV/Index Q,
and uses one warp for Index Q, without a new Index Q -> Score early trigger.

| Context | Layer | Existing QB PDL us | Fork + Index Q 1 warp us | Reduction us | Reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 32768 | FULL2 | 241.664 | 239.616 | 2.048 | 0.85% |
| 32768 | FULL1 | 268.312 | 266.128 | 2.184 | 0.81% |
| 131072 | FULL2 | 385.016 | 382.952 | 2.064 | 0.54% |
| 131072 | FULL1 | 464.936 | 461.992 | 2.944 | 0.63% |

These small differences overlap capture-to-capture variation. They are not an
end-to-end TPS claim or proof of a globally optimal schedule. The extra gain
is mainly an ordinary fork/resource change, **not evidence that broader PDL
itself helps**. The new candidate is not promoted to the model implementation.

Validation: 32 existing tests plus 60 Index Q boundary-test invocations passed;
the latter cover signed zero, FP4 midpoints, subnormals, negative positions,
ragged selector initialization, input immutability and changed-input Graph
replays. Every accepted full-layer capture also checked outputs, selections
and all cache/state pools byte-exact against serial execution. A separate
32-trace pass verified the 29/30 FULL2/FULL1 kernels per replay. Profiling
was never active in the performance samples. WO_A quantized epilogue remained
unavailable in this runtime, as in the earlier experiment.

Two benchmark issues were found and corrected in the portable runner:

1. `torch.cuda.Stream()` returns round-robin pooled handles, even while earlier
   wrappers remain alive. After enough captures, a new caller could alias the
   cached global/auxiliary stream and silently serialize a branch. Capture now
   excludes those handles and saves/validates stream identities. A 40-capture
   regression crosses the 32-stream pool boundary.
2. This runtime's `cuda-bindings` edge-data views sometimes contain corrupted
   first entries. The audit now owns the `cuGraphGetEdges_v2` output buffers.

A graph-rewrite prototype was rejected by CUDA with error 801: dependencies
cannot be removed from a graph containing allocation/free nodes. The working
fork is constructed during stream capture, without deleting allocator edges.
The experiment retains kernel tensor arguments and checks QKV/Index Q buffer
conflicts. Porting it requires equivalent ownership checks in the engine;
do not apply a post-capture edge deletion to the production graph.

## Next PDL Work: Borrow the Mechanism, Not Just the Switch

The pdlrt and GLM5.3 studies show a broader producer/consumer protocol than the
current one-edge experiment. This handoff includes no private implementation
files or vendor code; the relevant mechanisms are:

- **pdlrt:** explicit programmatic graph edges and selective chain/fork support;
  avoid leaving a redundant ordinary completion edge on the same dependency.
  Independent weight/KV prefetch can run before the consumer wait. Some paths
  separately publish partial results with release/acquire readiness protocols.
- **GLM5.3:** coordinated host launch attributes, device trigger/wait, and
  programmatic events between branches. Specialized GEMM loaders separate
  static weights/scales from dependent activations. QKV post can prepare
  positions/slots and norm weights before waiting for projections. This is
  more than enabling DeepGEMM's global policy.
- The studied GLM RTP integration had a small-M gate (M <= 256); DSv4.1 B64/S6
  is M384. Its group128 scaling also differs from this path's group32 layout.
  Do not transplant the kernel or extrapolate small-M gains. Framework TopK and
  FlashMLA calls were not automatically replaced with PDL kernels everywhere.

Useful navigation in separately available reference checkouts:

| Reference snapshot | Read first |
| --- | --- |
| pdlrt `9af2b467fc01` | `pdlrt/graph.py` PDL whitelist/chain forks; `pdlrt/cubin.py` explicit programmatic edges |
| Same pdlrt snapshot | `splits/pure_mla_seq4/a1q_aproj.py` independent weight loader; `splits/sparse_select_seq4/sq6_index_core4.py` KV prefetch before wait; `splits/pure_mla_seq4/a6q_attn.py` attention prefetch protocol |
| RTP GLM study `e23733ee65b8` | `glm5_cmp.py`: stream topology and programmatic events |
| CUDA extension study `cf98e119c02d` | `csrc/glm5/include/fp8_gemm.cuh` split loaders; `bf16_gemm.cuh` trigger/wait placement; `ops/qkv_a_proj/kernel.cu` independent position/slot setup |

Those snapshots explain the design; they are not dependencies fetched by this
runner. The GLM sources require access to their owning repositories.

Three different problems must stay separate:

| Relationship | Appropriate mechanism |
| --- | --- |
| No data or buffer-lifetime dependency | Ordinary Graph branches/multistream |
| Dependent computation, independent initialization/prefetch | Producer trigger + consumer PDL wait + correct launch/graph protocol |
| Only some rows/blocks are ready | Split scheduling or explicit ready/epoch protocol; not automatic PDL |

### Ordered Plan

1. Complete the four-policy 32K/128K controls on the destination, with at least
   two independent processes and multiple captures. Confirm correct dispatch,
   bitwise results and actual graph edge types before adding more changes.
2. Reproduce enqueue-order improvements independently. Keep event-record points
   tied to actual data readiness; do not accidentally change a submission hint
   into a completion wait. Retain a selectable unmodified baseline.
3. Extend **FULL-owned** producer/consumer protocols one subgraph at a time:

   | Candidate chain | Independent work to investigate | Main constraint |
   | --- | --- | --- |
   | Input quant -> QA -> norm -> merged QB | Weights/scales and launch setup before activation wait | Existing GEMM wait/load placement may limit benefit; do not modify DeepGEMM |
   | QB + Indexer weights -> Index Q post -> Score | RoPE positions, metadata, read-only KV prefetch | Two producer parents; cross-stream programmatic events require a complete protocol |
   | V/G -> Global A -> Index K -> Global B | Positions/slots/norm weights and projection weights | FULL2 old-ring read before state overwrite; Global/Index cache visibility |
   | Score -> compact -> token/candidate selectors -> finite/slots | Selection setup and metadata | Scratch ownership; TopK/native dispatch may not support a consumer wait |
   | MLA -> WO_A/quant -> WO_B | Output-projection weight preparation | Third-party producer support and actual fused/unfused WO_A dependency contract |

   These are investigation targets, **not implemented PDL edges**. Replacing a
   vendor GEMM with an RTP-owned specialized kernel is a separate design choice
   requiring full numerical/layout/shape validation, not a hidden vendor patch.
4. For every proposed edge record producer input/output pointers, RAW/WAR/WAW
   hazards, scratch ownership, trigger location, consumer wait, independent
   prework, graph edge/launch attributes, and resource occupancy. Inspect generated
   PTX/SASS, not just Python/CUDA source: compiler scheduling can move loads.
5. Consider grouped TopK -> MLA only after whole-kernel PDL is stable. A per-row
   ready flag needs release/acquire ordering, replay epochs/reset, cache visibility
   and a progress argument. PDL does not guarantee concurrent residency; spinning
   consumers must not occupy all resources needed by their producer. Start with
   explicit grouped completion nodes if a simpler safe design suffices.
6. Accept only reproducible **full Graph wall-time** improvements with unchanged
   correctness. Keep per-edge microbenchmarks for diagnosis, not as a substitute
   for the whole-layer result. Then separately validate full-model execution.

### Debugging Procedure

1. Reproduce a failure with fixed seed, one shape, one capture, and preserved
   source/dependency hashes. First separate serial, multistream and PDL failures.
   A native import/JIT/ABI failure is not a numerical or scheduler result.
2. Test eager and capture/replay separately on a non-default caller stream.
   A successful first replay is insufficient: change inputs, starts, tables and
   state across replays. Include M=1/7/64/97/384/769/1024 for the chain, and varied
   B/S with FULL1/2 state-boundary cases for the full graph.
3. Compare the earliest divergent intermediate: Q, FP8/scales, Global output,
   Index K cache/state, Index Q, logits, token indices, candidate flags, slots,
   MLA and output projection. Check cache padding/invalid slots and nonfinite
   values rather than only the final output.
4. Inspect `*_edges.json`: an early trigger with an ordinary completion edge
   gives no intended launch overlap; a programmatic edge without the required
   consumer wait is unsafe. Do not blindly edit edges of an old serial capture:
   its allocator reuse can depend on the removed ordering.
5. Inspect cached PTX/SASS for `griddepcontrol.launch_dependents` and
   `griddepcontrol.wait`. Verify independent loads really precede the wait and
   dependent loads follow it. Check async-copy/TMA completion and proxy ordering
   as well as ordinary loads; comments alone do not prove memory visibility.
6. Profile separately. Check dispatch, kernel counts, registers/smem, grid sizes,
   actual waits, branch joins and overlap. PDL can overlap setup yet lose overall
   due to register/shared-memory occupancy or memory-bandwidth contention.
7. Recheck metadata-unfused and legacy-TopK fallbacks, unsupported layouts, cold
   capture, repeated shapes, exception joins and allocator pressure. Sanitizers
   can help find errors, but a clean report is not a proof of a cross-kernel
   readiness/progress protocol. Never accept a skipped test as coverage.

## Source Map

All paths below are in this repository; no original-machine path is required.

- [Runner](../../rtp_llm/test/perf_test/dsv41_full_decode_pdl.py)
- [FULL gate, streams, joins and PDL gate](../../rtp_llm/models_py/modules/dsv41/fp8/decode/full_overlap.py)
- [FULL integration and candidate branching](../../rtp_llm/models_py/modules/dsv41/fp8/attention_v41.py)
- [QKV caller](../../rtp_llm/models_py/modules/dsv41/fp8/decode/compute_qkv.py)
- [Q norm/quant producer](../../rtp_llm/models_py/modules/dsv41/_v41_query_quant.py)
- [Merged QB](../../rtp_llm/models_py/modules/dsv41/fp8/_v41_merged_qb.py)
- [Global producer and state handling](../../rtp_llm/models_py/modules/dsv41/fp8/_v41_decode_global.py)
- [Metadata and Index Q/Score](../../rtp_llm/models_py/modules/dsv41/fp8/_v41_decode_indexer.py)
- [Token/candidate selection](../../rtp_llm/models_py/modules/dsv41/fp8/_v41_decode_topk.py)
- [Fixtures and graph regressions](../../rtp_llm/models_py/modules/dsv41/fp8/test/test_v41_global_overlap.py)

Keep local traces, checkpoints, credentials, native binaries, environment dumps
and machine-specific launch wrappers out of the public branch. Preserve failed
attempts locally and start a new output directory for each corrected experiment.
