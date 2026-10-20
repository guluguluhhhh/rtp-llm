# V4.1 QB/KV 融合实施与验收记录

2026-10-20。实现分支 `codex/dsv41-qkv-integration-review`，代码提交
`50dbe007d8ecfdc75230dd99c6d6725e13e9d8b8`。已从原评审基线 `d6826d6`
rebase 到 `guluguluhhhh/full_layer_optimize` 的
`801eb19c34c4616c8ed387aafada185f879e00b9`。

本次完成可单独开启的 FULL producer 接入：**merged QB + SWA，2→1**，以及
**TopK finite publication + slot conversion，2→1**。两个区间在正反顺序复测中均有
收益。新功能默认关闭；整包构建、整层准确性和三流 E2E 尚未验收。

## 实现范围

- 新增 native backend、Bazel target 和 Python dispatch，生产调用无需实验 JIT loader。
  直接使用 `[36864,1280]` 的共享 merged weight、packed SF 和既有 raw KV view。
  Main Q / Index Q 保持同一输出 buffer 的 views，token row stride 为 36864。
- Q 留给 Mega consumer 做 RoPE；side warps 做 KV norm、RoPE、BF16 materialization
  和 528B cache 写入。支持 int32/int64 positions/slots、padding page 和负 slot。
- Qnorm→QB 的 incoming PDL 显式传参，读取任何 payload 前 GDC wait；没有引入
  producer→FlashMLA outgoing PDL，也没有更换 FlashMLA backend。
- Slot fusion 保留 shared local TopK，额外返回当前 forward 的 physical slots；
  不支持的布局沿用原转换。FULL 的候选/Sparse Plan stream 调度保持原有调用关系。
- 新 head 中 Reindex 已有 front projection / QB query-preparation 融合。
  新 QB+KV decode 分支显式排除 Reindex，并测试保留其 `head_weights/index_prepared`。

FULL 的当前 TopK 依赖 merged QB 输出的 Index Q，所以不能放回这次 QB launch。
本次没有接入 REUSE ready-index **3→1**，不能把上述两段相加宣称为原始 3→1 或整层收益。

| 控制项 | 默认 | 行为 |
| --- | --- | --- |
| `DSV41_CUSTOM_QB` | 0 | pure merged-QB 性能对照；正常融合接入无需开启 |
| `DSV41_FUSED_QB_KV` | 0 | FULL QB+KV side workers |
| `DSV41_FUSED_TOPK_SLOTS` | 0 | FULL batched paged TopK 后处理与 slot mapping |

初始支持 M128/192/256/512，M 是 B×S。M128/192 使用 swap BM128/BM192、72 clusters；
M256/512 使用 non-swap BM256/BN256、单 accumulator stage、72 clusters、4 KV warp/CTA。
M384 的新 N36864 pure-GEMM 候选尚未达到基线，继续 fallback。

## 验证环境与方法

- SSH `sm103-0`，PyTorch runtime capability `(10,3)`，148 SM。
  PyTorch `2.11.0+cu130`，nvcc `13.2`；共享 GPU、未锁频。
- 目标 wheel `deep_gemm-2.8.2+ad62a90.cu132`，下载 SHA256：
  `6ca06c8b60ee4eeb0aaa9d989987d219c2c468b971b558120e95a24a89f3ef07`。
  最新分支 `0001-device-gemm-entry.patch` 已在隔离依赖副本应用，patch/header SHA
  存入随附 JSON；未替换全局安装。
- native TU 使用仓库锁定的 CUTLASS `dc4817921edda44a549197ff3a9dcf5df0636e7b`
  构建。旧 header 的 SM103 TMA 与 TCGEN 兼容 guard 已补齐；不启用时会分别
  assert / `brkpt`，不能只检查编译成功。
- 数值与测试方法锁定官方 DeepGEMM testing
  `559d79fb6994a58b8a15b4b93bf13ccc16edf247`，检查 origin、HEAD 和文件 clean 状态。
- FP8/SF 由目标 DeepGEMM utils 构造；Q 对 FP32 reference 使用官方
  `calc_diff < 1e-5`，并要求与目标 DG bitwise 一致。
- 性能先完成 3 次 callable warmup、Graph capture 和 20 次 replay warmup。
  每个样本 `8GB flush → 1 Graph replay → fence`，30 samples，取中位数。
  Trace 审计 kernel 数和签名；两 kernel 区间使用 `max(end)-min(start)`，
  一 kernel 区间同一包络等于其 activity duration。无 event dispatch 替代或 kernel median 求和。

## 正确性

QB/KV suite **7 tests，0 failures/errors/skips**；finite/slots suite
**3 tests，0 failures/errors/skips**。

覆盖四个支持 M、int32/int64 metadata、strided raw KV、padded pages、negative slots；
同地址修改 FP8/raw KV/positions/slots，poison 输出后 Graph replay，Q、Index Q、BF16 KV
和 cache bytes 逐项精确一致。另覆盖 Qnorm incoming PDL on/off 的 changed-input replay、
实际 `decode_compute_qkv` 的 B128×S1/B64×S2 接线、Reindex prepared 路由和 unsupported fallback。
Slot 测试覆盖 K512/1024/2048、FP32/BF16 logits、entries64/128、短行、NaN/±inf、
非顺序 request ids、padding block table、base<=0 和同地址更新后 replay。

没有放宽 KV 容差。最初发现的少量差异通过对齐 Triton `blocked<4,32,4>` 的归约顺序、
rsqrt 和 RoPE imaginary FMA 操作数顺序修正。合法 size-1 维的任意 stride 也已纳入门控测试。

这些是**精确 TU + test bridge 的 GPU 测试**。机器现有 RTP wheel 缺少新分支的
`HiddenStateCaptureDtype`，无法完整 import 当前 RTP package；query/front/selector
测试仅隔离了依赖导入，使用原文件未改写的函数 AST。投影/原生 TopK 的接线测试使用
确定性 stand-in；真实新 kernel 与原 Triton oracle 均在 GPU 执行。这不等同于完整模型验收。

## 两段融合的配对结果

以下均为各自生产区间的 Graph kernel envelope，单位 μs；不含 attention，PDL=false。
轮次 A 正向 shape/候选顺序，轮次 B 反向。保留共享 GPU 波动，不把最小值当中位数。

| M | QB+KV A：2→1 | QB+KV B：2→1 | 时长降低 |
| --- | --- | --- | --- |
| 128 | 18.209 → 15.104 | 19.744 → 16.256 | 17.0%–17.7% |
| 192 | 19.328 → 15.920 | 19.424 → 16.000 | 17.6% |
| 256 | 19.840 → 16.800 | 19.648 → 16.848 | 14.3%–15.3% |
| 512 | 29.136 → 25.392 | 27.905 → 24.800 | 11.1%–12.9% |

| M | finite+slots A：2→1 | finite+slots B：2→1 | 时长降低 |
| --- | --- | --- | --- |
| 128 | 4.992 → 3.568 | 5.664 → 3.664 | 28.5%–35.3% |
| 192 | 5.440 → 3.808 | 5.665 → 3.808 | 30.0%–32.8% |
| 256 | 6.047 → 5.216 | 5.889 → 4.000 | 13.8%–32.1% |
| 512 | 7.808 → 6.672 | 7.424 → 5.152 | 14.5%–30.6% |

这些 isolated region 结果证明了减少 launch 的收益，尚不能判断 Global/Index-K、
candidate/Sparse Plan 以及 MLA 同时运行时的总收益。

## Pure GEMM 复测与非平稳样本

M128 在两轮中 custom 慢约 0.3%/2.9%；M192 快约 0.6%/0.8%；M256 慢约 1.7%/1.6%。
M512 的整 sweep 原始中位数则是 A `28.128→29.568`、B `24.480→27.536`，
不能将这两组结果描述成 pure GEMM 已持平。

检查 30 个样本发现同一 Graph 的时长在采样过程中发生明显变化：例如 B 的 custom
从约 28 μs 降到约 25 μs；A 的 DG 也从约 32 μs 降到约 27 μs。未改 kernel、
binary、输入和测量方法，追加 M512 独立正反顺序配对三轮：

| 配对 | DG | Custom | Custom 额外时长 |
| --- | --- | --- | --- |
| A | 24.240 | 24.608 | +1.52% |
| B（反向） | 24.240 | 24.576 | +1.39% |
| C | 24.401 | 24.784 | +1.57% |

独立复测的差距接近此前 1%–2% 的结果。原 sweep 的波动与非平稳 GPU 执行状态一致；
没有时钟/资源硬件计数器，不能进一步断言是 DVFS、其它任务或某个缓存效应。
全部异常轮次与独立复测均保留。生产收益仍需在可控环境做整层验收。

## 带宽

另测 1GB FP32 streaming read/write kernel（`add zero`），实际流量约
**1GB 读 + 1GB 写**。Graph 单 kernel body 中位数 **291.779 μs**，
读写合计有效带宽 **6.85 TB/s**，30 samples，结果精确。
它不是 memcpy engine 带宽、GEMM 的 HBM counter 或硬件峰值。

## 复现与剩余验收

在与此源码匹配的 CUDA13 RTP 构建环境运行：

```bash
bazel test --config=cuda13 \
  //rtp_llm/models_py/modules/dsv41/fp8/test:test_v41_qb_kv_x86 \
  //rtp_llm/models_py/modules/dsv41/fp8/test:test_v41_topk_slots_x86
```

准备锁定官方 testing checkout 后，匹配 native 库可直接运行 portable runner：

```bash
python -m rtp_llm.test.perf_test.dsv41_qb_kv \
  --testing-root "$DG_TESTING_ROOT" --out results_qb_kv_a --bandwidth
python -m rtp_llm.test.perf_test.dsv41_qb_kv \
  --testing-root "$DG_TESTING_ROOT" --out results_qb_kv_b \
  --batches 512 256 192 128 --reverse
```

当前机器缺少匹配的新 RTP native wheel 和 Bazel，尚未做整包构建、完整三流 decode、
整层准确性、真实模型权重 reload 和其它 GPU/TP/CP 配置验收。REUSE ready-index 3→1
另做增量；这些限制是保持新开关默认关闭的原因。

随附 [JSON](dsv41_qb_kv_integration_results.json) 保存源文件/binary/dependency hashes、
每轮统计、每个 sample 的包络、raw metrics/trace SHA。各 kernel 的边界、完整
原始 trace、测试 harness 和日志保留于：

- 本地 `experiments/dsv41_q_attn_fusion/integration_v10/`。
- `sm103-0:/home/zhouzhenxiang.zzx/dsv41-official-baseline-20261019/integration_v10/`。

最终采用 `results_latest_patched_a/b` 和 `results_m512_qual_a/b/c`；先前失败或旧依赖的
诊断目录不作为本表数据。GPU 未运行的 Bazel targets 不标记为通过。
