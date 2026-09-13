# 用状态未更新的红测识别 recurrent KDA 图 padding 漏写

2026-07-23，AscendC recurrent KDA 替换验证中，最小 A3 用例使用 TP16 的局部形状：六个 head、维度 128，图容量 16，但实际只有一个 token。`cu_seqlens=[0,1,1,…]` 描述一条有效序列和零长度占位行，不能把最后一个累计偏移 1 当作 allocation 必须填满到 16 的承诺。

旧 `ValidateCuSeqlens()` 已检查累计偏移单调、每段长度合法、末端不越过 T，最后却还要求 `last_offset == T`。这使合法 padded 输入在逐行计算前整体返回。最小修正保留边界检查，解除“有效长度必须等于图容量”的条件；逐行长度仍按 `cu[i+1]-cu[i]` 求出，零长度行在访问 state metadata 前跳过。

测试没有把 padding 为零当作主要成功条件，而是分别核对：有效输出与参考一致、active state slot 被正确更新、其余 state 不被误写。原红测中 state 有 49,583 个元素不匹配，最大绝对差约 0.564；修后 clean build 的七项 NPU 用例通过，包含多步状态更新、accepted 长度 1–8 和图回放，另有八项源码/契约用例通过。“15 passed”是两类测试合计，不是十五项 NPU。

九月还审查了另一件事：wrapper 是否必须在 norm-gate 前后各做一次 mask。即使 kernel 不写尾部，gate 中的 NaN 仍可能经 `0 * NaN` 产生 NaN，因此“中间已经补零”不证明最终尾部有限。但删掉前一次 mask 是否可行没有独立删除式图回放验收，不能写成已验证优化。

这两次调查分别涉及有效行被整体漏算，以及最终 padding 清理责任。引用时应保持这一区分，不能把容器健康、尾部全零或 mask 数量直接当作状态更新正确的证明。[历史算子工作](https://github.com/vllm-ascend/vllm-ascend-kimi-k3/pull/10)。
