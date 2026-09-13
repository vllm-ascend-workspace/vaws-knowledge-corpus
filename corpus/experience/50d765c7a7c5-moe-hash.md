# MoE Hash 调查：功能同名不等于实际链接了同一接口

2026 年 9 月，定位 DeepSeek V4 专家路由算子时，先在 ops-transformer 找到 `moe_gating_top_k` 目录，容易据此认定模型已经使用上游 V2。继续从 Python router 追到 C++ binding，才确认当时 sqrtsoftplus 路径的实际调用链：

```text
router 的 sqrtsoftplus 分支
  → torch.ops._C_ascend.moe_gating_top_k_hash
  → C++ binding
  → aclnnMoeGatingTopKHash
```

该分支中，有 `tid2eid` 映射表时要求同时传入 input IDs，调用前还将 IDs 转成 INT64、映射表转成 INT32，并按通信和 token 分片路径对齐行数。没有映射表时传入空的可选参数。也就是说，走到名为 Hash 的自带符号，仍不等于本次输入一定启用了按 ID 查表的专家选择。

调查将两个容易混淆的概念拆开：Hash 决定如何选择专家，sqrtsoftplus 决定如何计算分数。历史上游接口中，V1 已有归一化类型参数，但其 ABI 不带 input IDs 与映射表；V2 增加这些可选输入，二者都提供时才进入 Hash 模式。不能从 `norm_type` 或目录名称推断可选输入、实际链接符号和完整能力。

最终保留两份映射：公共算子的参数与功能，以及模型实际的 Python/C++/ACLNN 消费链。这样，自带 Hash 的设备测试就不会被误算成上游 `aclnnMoeGatingTopKV2` 的支持证据，某个上游版本的架构限制也不会被直接套到另一个实现。

本次是公共源码调查，没有执行上游 V2 的独立 NPU 验收。继续使用这项经验时，应确认当前安装包实际导出的符号和版本，再查该符号对应架构的参数契约；历史接口表本身不是安装环境保证。
