# Compressor metadata 复用：语义组、子步生命周期与消费者取证

2026 年 8 月，减少 Compressor metadata 的跨层重复计算时，首版按对象身份和压缩比缓存。问题是同压缩比的两组缓存可能使用不同物理 block table；同一对象也可能在合并 MTP/DSpark 前向中更新 `start_pos`、长度与 RoPE 输入。缓存命中不等于复用条件成立。

当时先改为由 metadata builder 提供稳定的语义组标识：同一 attention group 的多个层名使用该组的规范前缀，而非 tensor/object 地址。缓存放在当前 `ForwardContext.additional_kwargs`；算子仍在模型 forward 中执行，便于 ACLGraph 捕获。后续取证版本进一步用 `(cache_group_key, type(metadata))` 区分同组的 metadata 类别。其生命周期可提炼为：

```python
# 历史设计的简化示意，不是可直接调用的现行接口。
for substep in composite_forward:
    cache.clear()  # 新子步可能改变位置、长度、slot 和 RoPE
    for layer in substep.layers:
        key = (layer.metadata.cache_group_key, type(layer.metadata))
        if key not in cache:
            cache[key] = compute_metadata(layer.metadata)
        consume(cache[key])
```

真正的共享条件不只在 key 字符串上。原始计算依赖完整 RoPE 表、`query_start_loc`、`start_pos`、`block_table`、物理 block 大小、slot 表示、压缩比、压缩输出容量和实际请求数；这些条件不一致时，不能借同组之名共享。回归分别构造同组的不同 metadata 对象、不同组和子步 reset，检查调用次数与返回对象；物理布局用例还区分 logical block 几何和实际 `storage_block_size`。

后来的精度排查没有只比较缓存对象。图内 probe 在首次生成时保存输出快照，在命中时同时保存缓存值和重新计算值；到真正的 Compressor 入口，再 clone 实际消费的 cos/sin、起始位置和累计长度。分析同时比较 miss/hit 的输入、首次快照与命中值、命中值与重算值、consumer 参数与预期值。这样能够分别识别 key 错、输出被覆盖、输入变化后未失效，以及接线拿错对象。大的张量有采样路径，不能把所有比较都说成全量逐元素覆盖。

原始 replay 记录的 `differences` 和 `consumer_differences` 均为空，离线检查也未在所读样例中发现位置、slot 归属或 RoPE 参考差异。这排除了这些已覆盖复用点，却没有闭环后来 65K 长输出的精度问题。该经验的可复用部分是语义与生命周期分层，以及一直检查到实际消费者；旧缓存实现不是要求当前版本重新采用的方案。
