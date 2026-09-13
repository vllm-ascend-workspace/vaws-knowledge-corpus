# SiTU 接入中，纠正了 routed 和 shared 的 INT32 含义

2026-07-22 至 07-25，Kimi K3 的 SiTU 量化接入初期混淆了三种数据：INT4 权重装在 INT32 容器中、GMM 数值累加器为 INT32，以及已经反量化的 BF16 激活。调查沿 Python caller、Torch adapter、host/tiling 和 kernel 核对真实数据语义，不能从容器 dtype 直接推出下一算子应该如何解释每个元素。

当时 A3 两条路径的处理不同：

| 路径 | 实际输入 SiTU 的数值 | 处理边界 |
| --- | --- | --- |
| routed WeightNz GMM1 | 已反量化 BF16 | 保留 GMM 的 BF16 输出；SiTU 做激活与动态 INT8 量化 |
| shared | 数值 INT32 accumulator | 先数值转换成 FP32，并使用对应反量化尺度，再做 SiTU/量化 |

shared kernel 的早期实现与测试都曾把 INT32 当作 FP32 位模式。这不是只改文档就能解决的问题：该轮最后一次修正虽然仍用 `ReinterpretCast<float>()` 取得可复用的 local storage view，但随后必须执行真正的 `Cast(xLocalF32, xLocal, CAST_NONE, ...)`。前者描述目的缓冲区，后者才将整数数值转成浮点；不能看到 view cast 就省掉数值 cast。

相应源码契约测试也从“INT32 分支只做位解释”改为检查两种输入都经过数值转换。因此撤回了“shared 也必须改为 BF16”的早期建议，并保持 routed/shared 在各自真实 GMM 输出语义上接入。测试本身若只固化旧实现的字符串，同样可能把错误假设当 golden。

原始记录包括 77 项框架回归，以及该轮最后一次 INT32 修正的源码契约检查。新 kernel 的同步和构建成功不能继承旧二进制的数值验收，更不是 A5 设备证明；本篇只记录当时 A3 数据契约被纠正的过程，不将所有后续版本的分派写成固定规则。[历史算子接入](https://github.com/vllm-ascend/vllm-ascend-kimi-k3/pull/16)。
