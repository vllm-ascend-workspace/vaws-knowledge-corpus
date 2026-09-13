# MLA 算子替换调查：相近计算图仍有 epsilon、布局和输出契约差异

2026 年 8 月对比 CANN 9.1 的 MLA Prolog V2 与当时 vLLM-Ascend 自带 MLAPreprocess 时，两者都出现投影、RMSNorm、RoPE 与 cache 写入。调查先确定 `torch_npu.npu_mla_prolog_v2` 的 wrapper 实际连接 `aclnnMlaPrologV2WeightNz`，再逐项对齐输入输出，没有直接互换函数名。

原始源码与接口读回已经暴露几项会影响替换结论的差异：

| 检查项 | 当时读到的具体差异 | 为什么会影响验证 |
|---|---|---|
| RMSNorm epsilon | 自带 kernel 内部赋 `1e-6`；Prolog V2 接口分别暴露 Q 与 KV epsilon，文档默认 `1e-5` | 默认参数不同即可造成数值差异 |
| KV 物理布局 | 自带测试区分融合 ND、独立 cache 及 NZ 模式；Prolog 接口有自己的 `cache_mode` | 相同逻辑 shape 不代表相同物理字节布局 |
| 量化参数 | consumer 传入投影反量化尺度、输入尺度、缓存尺度；Prolog 还有各路可选参数及 Q NoPE 输出尺度 | 不能仅比较主输出而忽略 scale 的归属 |
| 输出与原地写 | Q NoPE/Q PE、KV 与 RoPE cache 更新各有位置；某些 cache 输出本质是原地更新 | 返回一个 tensor 或调用成功不能证明目标 cache 已正确更新 |

其中 NZ 测试明确按 `[block, C1, block_size, C0]` 转成逻辑 `[block, block_size, C1*C0]`，还有首轴间隔和 guard 区检查。比较时应先还原各自物理格式，再对逻辑值与未写区域分别断言，不能直接拿不同 layout 的 flatten 字节做精度结论。

当时已有用例能检查写输出及部分布局，但没有执行两个算子的同输入逐张量数值比较，因而不足以证明可替换。可提炼的下一步是：固定权重预处理与量化配置，显式对齐两路 epsilon，分别比较 Q 分量、cache 实际写入及 scale，并检查原地别名。此处记录的是旧接口差分；后续生产入口已变化，不能把这张历史比较表当成当前可选算子清单。
