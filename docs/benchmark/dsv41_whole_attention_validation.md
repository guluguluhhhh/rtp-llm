# 无 Bazel 的完整 REUSE attention 验证

2026-10-21。源码基于 `749defccf9`，另补 producer PDL visibility 修复。

**已在 GPU 跑通完整 attention 计算链，未使用 Bazel。** 五档 M128/192/256/384/512，
另有 B64×S2、compression ratio2；每项三阶段同地址更新输入/metadata/权重。
每轮36组最终输出对照全部 `calc_diff=0、max_abs=0`，cache bytes精确一致。
15项回归测试（13 GPU、2 CPU）全部通过，无跳过。

## 实际执行边界

入口是生产 `_forward_decode_body` 的 `[B,S,5120]` BF16输入，执行以下真实实现：

1. Position conversion / RoPE table lookup。
2. 原生group32 v3输入量化、真实 `V41MXFP8Linear` QA/KV GEMM（N1792/K5120）。
3. 真实weighted Q norm/FP8 quant（latent1280）。
4. 原DG Q GEMM + Triton KV norm/RoPE/cache + slot translator，或新融合producer。
5. 原MegaAttention decode调用/新native PDL consumer。
6. 真实grouped WO-A GEMM、group32 v3量化、WO-B GEMM，返回 `[B,S,5120]`。
7. 生产TP1 output collective分支（无需通信），完成attention的forward-local bookkeeping。

前后projection、quantizer、attention和output kernel都没有mock。输入为随机有限checkpoint
weights、真实GPU cache bytes、ready TopK；fixture仅提供KVCache的CPU元信息/存储接口。
这验证的是 **完整attention子层**，未覆盖Transformer的MHC/residual/MoE或完整模型。

旧RTP native wheel无法import新分支的完整配置包，因此通过未改写的原函数/方法AST隔离
constructor/config导入；source SHA、选取AST SHA和继承边界均记录在JSON。
实际 `_forward_decode_body`、pool view、slot mapping、projection等method body保持原样。
这不验证生产constructor、整包链接、真实checkpoint loader或服务启动。

目标分支锁定的 DeepGEMM2.8.2+ad62a90不提供`epilogue.FP8Group32`和
`get_jit_config_fingerprint`。本轮正常执行源码提供的WO-A BF16→v3 quant fallback；
baseline和fused均为此路径，不将它标为WO-A融合量化路径的性能。

## 本轮发现并修复的协议缺口

原producer以`wait_pdl`决定是否GDC wait，但以`wait_pdl || trigger_pdl`开启launch属性。
在完整链中，outgoing PDL使Qnorm→producer也成为programmatic edge；
即便Qnorm不显式trigger，kernel exit本身仍不能替代该边上的global visibility wait。
现改为 **`side.wait_pdl = wait_pdl || trigger_pdl`**，任何开启该属性的producer都会在
读取输入前等待。此问题由Graph边审计和CUDA协议核对发现，未宣称已复现数值错误。

完整PDL-on Graph为 **10个kernel节点、2条programmatic edges**：
`Qnorm → producer → FlashMLA`。baseline是12节点/0条，fused-off是10节点/0条。
之前孤立的producer/consumer两节点图仍是1条PDL边，边界不同。

新增GPU回归测试覆盖非PDL Qnorm作为真实上游producer，M128/512的changed-input replay，
同时审计3节点/2条programmatic edges，最终FP8输出、SF和LSE精确一致。
协议依据：[NVIDIA Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)。

## 基线、数值与依赖

这次baseline使用目标分支锁定的 **FlashMLA1.0.0+2e5429f** wheel，SHA256：
`047ea42f62f8a7057dfcf376f521aa53ecc5d7d1731b9491cc75b70863e96ce1`。
仅在隔离目录解压，未替换全局安装。PDL-on使用仓库固定的0f11e178 native consumer。
PDL-off使用同一目标wheel；baseline/两种fused的完整层最终输出和cache均已比较。

- Qnorm producer、native Q/KV/index、FlashMLA consumer和v3 quantizer都实际执行。
- 每阶段修改X、positions、req、block table、local TopK；最后阶段同地址修改Q权重。
  poison最终输出后Graph replay，检验动态值可见性，local TopK保持只读。
- 两轮各36组最终输出对照，`calc_diff=0、max_abs=0`；cache bitwise一致，输出全finite。
- 数值函数取自干净的官方DeepGEMM testing
  `559d79fb6994a58b8a15b4b93bf13ccc16edf247`，校验origin/HEAD/clean状态。
- runtime PyTorch CUDA capability `(10,3)`、148 SM；torch2.11.0+cu130、nvcc13.2。
- 重新编译fix后的精确native TU，15项回归测试（13 GPU、2 CPU）0 failures/errors/skips。

## 完整 attention 性能

两轮使用相同的各路径源码/binary和inputs/weights；A正序、B反序。3次warmup、capture一次完整调用、
20次replay warmup；30样本，每样本`8GB flush → 1 Graph replay → fence`。
flush实际为4个kernel，独立signature审计后剔除，不能假定8GB zero只有一个kernel。
计量为CUPTI `max(kernel_end)-min(kernel_start)`，含position元信息及全部前后计算；
不含flush/JIT/allocator/host dispatch/fence。共享GPU未锁频。

| M | A：baseline / fused off / fused on（μs）| B：baseline / fused off / fused on（μs）| PDL-on 相对 baseline |
| --- | --- | --- | --- |
| 128 | 91.889 / 90.593 / 94.609 | 90.112 / 89.792 / 89.409 | -3.0%～+0.8% |
| 192 | 113.889 / 113.281 / 107.825 | 109.521 / 106.754 / 106.272 | +3.0%～+5.3% |
| 256 | 111.969 / 109.761 / 109.361 | 112.161 / 109.601 / 109.649 | +2.2%～+2.3% |
| 384 | 136.929 / 132.754 / 133.234 | 136.624 / 132.912 / 133.137 | +2.6%～+2.7% |
| 512 | 159.953 / 155.681 / 154.785 | 159.986 / 156.065 / 155.266 | +3.0%～+3.2% |

正号表示时长缩短，负号表示退化。M192–512在两轮PDL-on中均缩短2.2%–5.3%。
**M128无稳定PDL-on收益**：A退化2.96%，B缩短0.78%，不能写成全batch稳定加速。
保留全部样本；没有硬件计数器支持将波动归因于DVFS/L2/其它任务。
PDL也未在全部M上稳定优于fused-off，开关保留。

### 同一trace里的原3-kernel区间

| M | A：原3-kernel / fused off / fused on（μs）| B：原3-kernel / fused off / fused on（μs）|
| --- | --- | --- |
| 128 | 18.528 / 16.976 / 18.128 | 18.128 / 16.992 / 17.441 |
| 192 | 20.497 / 17.920 / 17.728 | 19.776 / 17.328 / 17.376 |
| 256 | 20.320 / 16.128 / 16.656 | 20.064 / 15.984 / 16.448 |
| 384 | 24.304 / 20.464 / 21.008 | 24.321 / 20.433 / 20.880 |
| 512 | 26.817 / 22.080 / 22.640 | 26.928 / 22.064 / 22.720 |

这是完整调用中实际Q GEMM→KV/cache→index的包络与单producer duration；
PDL-on producer的入口wait也在其duration内。不是单独测量median求和，也不是整层时长。

## 复现与材料

测试桥位于 `experiments/dsv41_q_attn_fusion/integration_v12/whole_attention.py`，
仅用于验证。复用v11的native loader，另编译精确的v3 quantizer TU；生产路径无JIT loader。
在准备好的容器执行：

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export RTP_QB_SOURCE_ROOT=/work/integration_v12/rtp-source
export PYTHONPATH=/work/integration_v10/dependencies/rtp_deepgemm_latest:$RTP_QB_SOURCE_ROOT
python /work/integration_v12/whole_attention.py \
  --wheel-root /work/integration_v12/flashmla_target --out NEW_A
python /work/integration_v12/whole_attention.py --reverse \
  --wheel-root /work/integration_v12/flashmla_target --out NEW_B
```

新回归测试归入`test_v41_ready_qkv.py`，可用精确TU测试桥运行，无需Bazel。
完整日志、CUPTI trace、per-sample metrics与失败的smoke诊断都保留在v12目录；
最终性能只用`results_a/b`。随附[JSON manifest](dsv41_whole_attention_validation.json)
保存全部样本、Graph边、源码/AST/binary/trace SHA及环境边界。

剩余验收是生产constructor/整包链接、真实checkpoint/full-model准确性、TP/CP和完整Transformer；
这些不由本次“无Bazel的完整attention计算链”结果代替。新总开关继续默认关闭。
