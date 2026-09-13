# GQA DSpark 的辅助特征取错残差流，模型仍能正常出字

2026 年 8 月的 Kimi K3／vLLM 0.27 迁移中，target 能启动、完成请求，并不能证明 draft 接线正确。GQA DSpark 的采信表现偏离既有运行后，调查从“draft 能否加载”转向了“target 在哪个位置交出辅助 hidden states”。

当时存在两种都具有正确形状、却不是相同数值的输入。Kimi 层间保留 raw prefix-sum stream；真正进入下一层 attention 的输入还要经过 Attention Residual mixer，将 prefix sum 与此前残差块混合。旧 Qwen3 GQA DSpark checkpoint 消费后者，MLA draft 则沿用上游的 raw stream。统一修改捕获位置会修好一种 draft，同时改变另一种 draft 的输入语义。

修复在 runner 中按 draft 配置选择模式，再通过 `set_dspark_aux_capture_materialized()` 传给模型。选择条件涉及 DSpark 方法以及 Qwen3 GQA 的模型类型、架构；没有把“启用了推测解码”直接等同于 materialized 模式。模型侧保留两个位置：

| 模式 | 当时的捕获位置 | 捕获内容 |
| --- | --- | --- |
| materialized GQA | 执行选中层之前，层号属于辅助层集合 | 用该层的 residual projection、norm 和有效残差块数计算 mixer 输出 |
| raw MLA | 执行上一层之后，以 `layer_idx + 1` 对齐辅助层集合 | 尚未经过下一层 mixer 的 hidden states |

回归测试没有只比较 tensor shape。它用可辨识的假层和 mixer：普通层输出增加 10，mixer 按有效块数增加 100；相同辅助层选择在 materialized 模式得到 `111`，raw 模式得到 `11`。这个构造能直接识别捕获早了一层、晚了一层，或把 mixer 漏掉的错误。另有 runner 分类及两层模型包装的参数转发测试，避免开关停在外层而未到达真正执行模型。

设备侧以 A3 四节点、DP4×TP16×EP64 的既有服务做有界复核。各节点 16 个 worker 均记录 materialized GQA 模式；固定同一份 8192-token 输入、请求 1024-token 输出，每轮先对四个 DP 实例重置 prefix cache，再向四个 DP 各发一次请求。第二轮四路成功，并检查了全 rank 错误日志。这支持该次 GQA 接线修复已进入实际运行；不能据此声称 MLA 也完成同样硬件矩阵，或把两轮请求的时延当成通用性能收益。

此案例对应当时的 [Kimi K3 迁移工作](https://github.com/vllm-project/vllm-ascend/pull/14454)。可复用的定位点是辅助特征的语义与层号约定：当 target 正常而 draft 异常时，需要同时追踪生成方的捕获位置和消费方训练时预期的残差流。
