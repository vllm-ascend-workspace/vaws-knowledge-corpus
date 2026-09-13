# FIA 非连续缓存验证：从错误测试入口追到丢失的 stride

2026 年 8 月的一次 A3 / CANN 9.1 适配希望 FIA 直接消费首轴非连续的 paged KV cache。早期的“两例通过”后来被撤回：合成 GQA 形状没有覆盖目标局部 head 关系，stock `torch_npu` 与自定义 OPP 的执行结果也被混在了一起。

重做验证时，先固定实际 `_C_ascend` 入口、custom vendor 的搜索优先级和加载动态库，再固定两种输入：

| 路径 | 当时实际构造的局部形状 | 需要保留的布局条件 |
|---|---|---|
| GQA | Q head 4、KV head 1、head dim 64 | K/V 的第 0 维有间隔，其他轴连续 |
| MLA | 局部 6 head 补到 8；latent 512、RoPE 64 | latent 同时作为 K 和 V；独立 RoPE 平面保持正常连续布局 |

原测试通过扩大 backing 后隔行取 view，形成首轴间隔；该构造可等价写作 `storage[slice(None, None, 2)]`。随后再在保持 stride 的情况下组织算子要求的 view。它检查非连续性，并使用非零逻辑 block。连续副本只用于对照，不能先对被测 view 调 `.contiguous()`，否则测到的是复制后的正确性。独立 FP32 reference 用相同逻辑 KV；MLA 比较还要按真实 head 数处理补齐部分。

红测最有区分力的现象是：MLA 非连续输出最大绝对误差约 0.771，而相同数据的连续对照约 0.002。由此把排查重点放到地址计算，而不是先改 softmax 数学。继续追踪得到如下传递链：

```text
Tensor descriptor 的真实 strides
  → host tiling.baseParams
  → arch22 MLA kernel 初始化 ConstInfo
  → paged K/V/RoPE offsetCalculator
```

修补涉及六个字段：`keyBnStride/keyN2Stride`、`valueBnStride/valueN2Stride`、`keyRopeBnStride/keyRopeN2Stride`。host 从 descriptor 读取第 0、1 维 stride 并保存；device 初始化必须把这些值复制进 `ConstInfo`，offset calculator 也要实际使用它们。只在 tiling 结构里加字段，或者仅打印出正确 stride，都不能证明寻址链已经接通。

补齐传递后，同组 GQA 与 MLA 用例通过；最大绝对误差分别约 0.00102、0.00196，连续与非连续输出的误差量级一致。这证明的是当时两种单算子构造下的正确性，没有证明库内部绝对无临时拷贝。历史工作见 [PR #13821](https://github.com/vllm-project/vllm-ascend/pull/13821)；迁移到另一版本时，应沿以上每一跳检查实际字段和调用入口，再复用红绿对照。
