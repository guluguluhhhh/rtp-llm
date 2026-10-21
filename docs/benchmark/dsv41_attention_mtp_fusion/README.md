# Attention producer + MTP fusion / Perfetto

Open `attention_mtp_fusion.html` for self-contained offline charts. To use the
embedded Perfetto UI, run:

```sh
python3 serve_report.py
```

Then open:
http://127.0.0.1:8769/dsv41_attention_mtp_fusion/attention_mtp_fusion.html

The report supports producer-only, producer + FlashMLA, full attention, PDL
controls, MTP proposal/commit, shape/dtype selection, and both measurement rounds.
Perfetto loads trace bytes through the official postMessage handshake; it needs
network access for the viewer and does not automatically publish traces. The
download button works offline; its JSON can be opened at https://ui.perfetto.dev.

`evidence.json` contains 40 attention and 140 MTP variant/round records with 5,400
single-replay samples. Raw Kineto traces remain in the shared workspace's
`experiments/dsv41_q_attn_fusion/integration_v13` and
`experiments/dsv41_mtp_fusion/integration_v14` archives. Source paths and SHA256 are
retained in this report. It includes the real representative sample activities
and their grid/block/register/shared-memory metadata, not the full raw archive.

`attention_mtp_overview.perfetto.json` contains eight independently measured paths
at M/B512, gamma5, round A. Its 81 GPU activities preserve relative starts and
durations; independent paths align to 1 us and overlaps use separate display
lanes. These synthetic groups are not simultaneous execution. Official Perfetto
TraceProcessor imported the comparison with zero errors. Browser QA exercised
160 selector combinations and opened the actual Perfetto UI both embedded and
in a new tab; see the JSON audits and screenshot in this directory.

Results cover attention and MTP preparation dependency subgraphs. MTP does not
include the full draft Transformer, head, or sampler; commit's intervening
feature projection/norm is excluded. The two preparation savings cannot be added
into a whole-round claim. PDL activity overlap can include GDC waits and does not
measure useful internal compute overlap.

See [MTP result details](../dsv41_mtp_preparation_fusion.md) and
[GPU reproduction](../dsv41_mtp_reproducer/README.md).
