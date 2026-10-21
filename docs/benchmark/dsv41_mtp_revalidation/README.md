# MTP push 后实测：bdb284c799

2026-10-21，在 sm103-0 原 GPU 容器，从本地已 push commit 的完整 git archive 建立独立源码目录后重跑。archive SHA256 `0215ab391d6a277213b3ddc15ac056ca5aa15f78f418fe7c6e96733d1b28e705`；生产源码和测试的 SHA 与提交逐文件匹配。没有复用旧结果，没有修改生产源码。

- 15 项 no-Bazel contract 测试：0 error / failure / skip；包括原 entrypoint 的融合分发、CUDA Graph replay、变化同地址 metadata、fallback。draft 测试 hook 仍是 unit-test double，不代表完整 draft 模型。
- 独立 kernel suite：5 项通过，已包含在上述 15 项中，不重复计数。
- B128/192/256/384/512，γ1/5/8，γ5 另测 int64 ids：35 配置、两轮正反序、140 variant-run、4,200 样本。
- 1,200 项 candidate 输出 tensor bitwise 检查通过，官方 calc_diff 全为 0；210 个 shape/phase case，含最终物理 slots。两轮输入 SHA 一致；与 v14 输入是否也一致：True。
- 官方 Perfetto TraceProcessor 独立审计 140 条 raw trace，逐 kernel 名称/数量/起止及 4,200 个包络匹配；允许 raw Kineto 已知的 overlapping-complete-event overflow 恢复警告，GPU 时间不能因此改变。
- 相比同轮原版，70 个 paired comparison 中退化数：0。core proposal 6/7→1，含原 metadata consumer 7/8→2；core commit 25→1，含原 ends cast/SWA consumer 27→3。

γ5、int32 ids，A/B=正序/反序：

| B | proposal 原版 A/B μs | proposal 融合 A/B μs | commit 原版 A/B μs | commit 融合 A/B μs |
| --- | --- | --- | --- | --- |
| 128 | 10.609 / 10.577 | 4.480 / 4.480 | 61.697 / 61.569 | 7.360 / 7.360 |
| 192 | 12.961 / 12.897 | 4.448 / 4.512 | 58.256 / 58.289 | 7.424 / 7.457 |
| 256 | 12.880 / 12.896 | 4.671 / 4.672 | 60.352 / 60.337 | 8.465 / 8.497 |
| 384 | 13.008 / 12.992 | 5.024 / 5.008 | 58.576 / 58.704 | 8.160 / 8.192 |
| 512 | 14.288 / 14.208 | 5.808 / 5.808 | 57.713 / 57.761 | 8.673 / 8.544 |

统一 metric：`exact_graph_kernel_envelope_us`。3 warmup、capture 一次、20 warm replay；每样本 8GB flush→一次 Graph replay→fence，4 个 flush kernel 排除。共享 GPU、未锁频。capability 由 PyTorch runtime 确认 `(10,3)`，148 SM，torch 2.11.0+cu130，nvcc 13.2。完整 raw trace 和逐样本记录保留在共享工作区 `experiments/dsv41_mtp_fusion/integration_v15_bdb284c799/results_a` 和 `results_b`，本仓库携带逐样本包络、trace SHA 与 Perfetto 审计。

## 完整一轮 MTP：本次尝试后仍被阻断

实际跑了直接 framework 测试导入和 `DeepSeekV41DSparkModel` 导入，二者均因已安装旧 C++ 库缺少 `HiddenStateCaptureDtype` 而失败，见 [实际错误日志](full_model_import.txt)。尚未执行三层 draft / lm_head / Markov / sampler。当前容器也没有 checkpoint mount。要测整轮，需要匹配当前分支的 C++ bindings 以及实际模型/服务 fixture；本次没有用简化浮点计算冒充整轮结果。

commit 子图不包含 actual caller 里的 feature projection / norm。proposal 与 commit 的节省不能相加成完整 MTP 收益。

[结构化结果](post_push_validation.json) · [运行阶段与命令](run_status.json) · [contract 测试日志](contract_tests.txt) · [独立 kernel 日志](kernel_tests.txt) · [Perfetto A 审计](results_a/perfetto_audit.json) · [Perfetto B 审计](results_b/perfetto_audit.json)

本次测试未修改生产源码。初次预检因容器 PATH 未包含 nvcc 停止；修正测试 driver 的 CUDA PATH 后，使用新的 run2 目录完整重跑。
