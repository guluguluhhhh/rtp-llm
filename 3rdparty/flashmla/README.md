# Isolated ready-QKV FlashMLA PDL consumer

`deps/git.bzl` pins the official DeepSeek FlashMLA source to
`0f11e178da1f67becb087a324b400854931ec3e3`, including its recorded submodules
(CUTLASS `4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b`). The producer separately uses
the existing RTP CUTLASS pin. Do not mix their header versions in one CUDA TU.

`0001-ready-qkv-pdl.patch` gives the selected kernel a private namespace and a
per-invocation PDL flag. It waits after barrier/TMEM setup, before any payload
read. Q loads use coherent global loads and producer-written indices use `.cg`
loads instead of `__ldg`; a grid dependency wait alone does not make `.nc` loads
a safe producer/consumer protocol. Launch uses explicit cluster and optional
programmatic-serialization attributes. The attention math is unchanged.

The RTP native wrapper instantiates only V4.1 decode, H64/D512, no Q norm,
FP8 SWA + FP4 extra cache, eight output groups and packed UE8M0 output scales,
for runtime SM103 / 148 SMs. It never calls the separately installed wheel's
parameter ABI or changes that wheel globally. With PDL disabled, the regular
MegaAttention path still calls the original wheel. Validation additionally
compares the patched native kernel with an unpatched wheel built from this
exact source revision; it does not establish parity with every FlashMLA wheel.

Both kernels are compiled ahead of time by Bazel; the torch extension loader
used for GPU qualification is a test bridge, not a production dependency.
Non-CUDA13/x86 builds expose an unavailable stub. See
`docs/benchmark/dsv41_ready_qkv_integration_results.md` for tests, source/binary
hashes, PDL-off/on measurements and outstanding full-package validation.
