# V4.1 MTP preparation reproducer

This benchmark covers integer preparation dependency subgraphs with the actual
metadata/SWA-slot consumers, not a full speculative round. See
[results and limits](../dsv41_mtp_preparation_fusion.md) and the
[interactive attention/MTP report](../dsv41_attention_mtp_fusion/attention_mtp_fusion.html).

`baseline_dspark_proposer_mixin.py` archives the pre-change implementation from
`438fbb2a0b8aa156b29184ee03daa774c64e7c25`. The benchmark extracts its unchanged
proposal statements and mapper AST; it loads candidate helpers and existing
consumers from `RTP_MTP_SOURCE_ROOT`.

On the original B300 container (runtime CUDA capability `(10, 3)`):

```sh
export RTP_MTP_SOURCE_ROOT=/path/to/this/rtp-llm
cd "$RTP_MTP_SOURCE_ROOT/docs/benchmark/dsv41_mtp_reproducer"
python benchmark.py --out results_a --modes baseline fused
python benchmark.py --out results_b --modes baseline fused --reverse
python perfetto_audit.py results_a
python perfetto_audit.py results_b
python run_contract_tests.py
```

Use fresh output directories. The original environment had PyTorch 2.11.0+cu130,
CUDA 13.2, and the framework's compiled Python bindings installed. Numeric checks
require a clean official DeepGEMM checkout at `/work/DeepGEMM-runtime-559d79f`,
origin `https://github.com/deepseek-ai/DeepGEMM.git`, HEAD
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`; the script verifies all three before
importing `deep_gemm/testing/numeric.py`. Install `perfetto==0.58.2` for the audit.
The no-Bazel contract runner uses real compiled `librtp_compute_ops` from the
original `/opt/conda310` environment; adjust that installation path on another
machine. It isolates module-level configuration imports, not function bodies.

`contract_tests.json` is the archived 15-test GPU result with source SHA locks.
Reproducer scripts are formatted for repository hooks; the archive retains the
original execution hashes. Candidate production/helper/test bytes match those
locks. The draft hooks in the existing unit tests are test doubles; their pass
does not establish whole-MTP numerical or performance validation.

Every performance sample follows 8GB flush -> one warmed CUDA Graph replay ->
fence. The metric is the exact CUPTI kernel envelope; flush kernels are excluded.
Raw trace archives remain in the shared workspace under
`experiments/dsv41_mtp_fusion/integration_v14/results_{a,b}`. The repository
contains per-sample envelopes, trace hashes, representative kernel activities,
and a Perfetto comparison trace; it does not contain the full raw trace archive.
