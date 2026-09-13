# 索引后处理融合：保持整数精度、可见边界和无效位置顺序

2026 年 9 月 10 日，一次索引后处理融合要把按得分返回的 TopK 索引变为 attention 消费的时间顺序。任务不只是排序，还包括按每个 token 的位置过滤不可见索引，并将无效位置放到末尾。参考逻辑在原始测试中为：

```python
visible = ((positions + 1) // compress_ratio).unsqueeze(-1)
valid = (selected >= 0) & (selected < visible)
sentinel = torch.iinfo(torch.int32).max
ordered = torch.where(valid, selected, sentinel).sort(dim=-1).values
result = torch.where(ordered == sentinel, -1, ordered)
```

当时 Ascend 原生扩展可用的是 FP32 sort。直接把 INT32 索引数值转换成 FP32 会合并 `2**24` 附近的相邻整数；融合代码改用可逆的位模式映射，把过滤后的非负索引和 sentinel 编成可排序的正常 FP32 数，再排序、解码。以下为原实现映射的提炼，所有运算先在 INT32 上进行，`bitcast` 只重解释位：

```text
SHIFT = 2**23
BASE = 0x81800000 - 2**32
bits(s) = BASE - s    当 s < 2**24
          s - SHIFT  其他情况
key = bitcast_fp32(bits)
# 对 key 升序排序后
s = BASE - bits      当 bits < 0
    bits + SHIFT     其他情况
```

低区间被映成负浮点键，高区间映成正浮点键，因此无需损失整数有效位。这里的输入域是过滤后的非负索引，`INT32_MAX` 被保留为无效 sentinel，最终也会还原为 -1；不是支持任意有符号 INT32 数据的通用排序器。位置的加一和整除仍遵循当时输入 dtype 的参考语义，不能把这次测试当作对溢出语义的额外修正。

kernel 对 TopK 向上补到二次幂，用 sentinel 填补列，并在写回时掩掉补齐位置。行分块同时考虑排序 scratch 的空间预算和 32 字节写入对齐；输入为非连续 view 时先形成连续输入，输出独立分配，测试检查原输入没有被改写。

88 项真实 NPU 用例通过：覆盖 TopK 1 到 2048 的选定规格、空输入、非连续输入、因果边界、重复项及全无效行；显式检查 `2**24±1`、接近 INT32 上限的值，并使用 INT64 位置让高索引在抽样测试中保持可见。它是全取值区间的抽样与边界验证，不是穷举全部整数。改变 selected 和 positions 后的图回放保持输出地址且精确匹配参考。与查询量化合跑的 131 项包含这 88 项，不能相加成另一批独立验证。
