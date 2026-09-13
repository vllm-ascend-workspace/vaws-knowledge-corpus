# GMM SwiGLU V2 迁移：撤掉隐式转换后才测到了原生 count 分支

2026 年 8 月，统一 GMM SwiGLU 算子时遇到 NZ storage shape 与量化数值两类差异。调查最初把 graph 问题解释为“V2 更依赖累计值”，于是 adapter 先对每专家 token count 做 cumsum，再强制传入 type 0。结果测试虽叫 count 用例，kernel 收到的始终是累计值，没有覆盖原生 count 分支。

重新检查 tiling 与 kernel 后，确认两种输入各有明确含义：

| group_list_type | 输入示例 | 第三个专家的 token 数 |
| --- | --- | --- |
| 0，累计结束位置 | `[1, 2, 2]` | 当前结束位置减前一结束位置，即 0 |
| 1，每专家计数 | `[1, 1, 0]` | 直接取当前元素，即 0 |

kernel 在 type 0 且专家编号大于零时才相减；type 1 直接使用列表元素。因此底层有两种模式，与适配层是否真正把两种模式传下去，是两个不同问题。后续沿 dispatcher、MoE 量化调用方、Torch adapter 和 C++ binding 删除隐式转换，列表内容与 type 一起传递；测试也检查实际传入的 tensor 和 type，防止再次出现“上层输入为 count，底层只测 cumsum”。空专家用例能直接暴露相邻累计位置处理错误，不能仅用每专家都有 token 的输入替代。

数值部分另行处理。融合量化保留原参考实现的标量倒数、逐行乘法和舍入顺序，不能因为公式代数等价就重新组合浮点运算；并纳入较大 token 规格，避免小矩阵掩盖分块差异。NZ storage shape 的架构差异也没有被 group-list 转换掩盖。

收敛后的 A3 原始记录为 16 例通过，覆盖 count/cumsum、空专家、单/多 Tensor、W8A8/W4A8 和 count 图回放。结束时部分 CI 停在 checkout/runner 阶段，未执行的设备测试不算通过。相关工作见 [PR #12894](https://github.com/vllm-project/vllm-ascend/pull/12894)。此次修复的关键是让输入真正抵达声称覆盖的原生分支；这组历史结果不随名称相同自动继承给后续提交。
