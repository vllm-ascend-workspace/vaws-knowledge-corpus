# 混合缓存改造中，用反向测试发现 group 语义被丢失

2026-08-07，一次旧 hybrid prefix-cache 改造对齐公共命中长度时，只裁剪了首个 full-attention group。最小反向测试构造两个 full-attention group 和一个 Mamba group：hash block 为 2 token，两个 full-attention block 分别为 4、8 token，输入 24 token；full-attention 已缓存完整输入，Mamba 只提供到第 6 token 的部分命中。

`find_longest_cache_hit()` 返回公共 hit length 6 后，测试期望各组所需 block 数为 `[2,1,2]`。旧实现实际返回 `[2,3,2]`，第二组仍携带过长命中列表。把同一测试先放回旧代码得到失败，再修正遍历相关 full-attention groups，定向用例及该文件四项用例通过。这样证明的是这条旧裁剪分支，而不是仅凭“现在测试绿了”推测过去有错。

2026-08-27 又遇到不同的分组重构问题：`UniformTypeKVCacheSpecs` 外层包装后，worker/scheduler 若只看包装类型或使用通用容量公式，就可能丢掉内部 Mamba spec 的 speculative blocks。调查沿两端分别追踪每层 spec，而不是只修一个 metadata builder。包装类的 block-table 宽度也要委托内部 spec 计算，并确认同一组各层所需宽度一致；直接按 `ceil(max_len/block_size)` 推导会漏掉特殊分片语义。

新增包装场景最初为 4 failed、1 passed；修后相关测试为 89 passed、2 skipped，没有重跑完整 NPU 服务。这个数字属于八月下旬的另一轮改动，不能与四项裁剪测试合并宣传成一次硬件验证。

后来的 main 对保留更长命中有例外，旧“全部裁齐”不能当作无条件规范。两次经历共同说明：重构 group 或 spec 容器时，既要检查外层分组结果，也要用反向测试证明内部命中边界、分片与容量语义没有被默认公式吞掉。[相关历史缓存工作](https://github.com/vllm-project/vllm-ascend/pull/13277)。
