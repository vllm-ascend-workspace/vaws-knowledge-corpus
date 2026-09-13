# 排查 Compressor 接口时，先分清逻辑布局、存储格式和 ABI

2026 年 8 月的一次 Compressor 接入调查中，讨论把 `TH`、模型 hidden size 和注册的 `FORMAT_ND` 混在了一起。真正需要回答的是：调用方传来的 RoPE 张量每个维度表示什么，以及这份张量以哪些参数抵达设备算子。

调查从 Python consumer 的 `_compute_metadata` 开始。原代码把 `full_compress_cos/sin` 用首维与末维整理成二维，再调用 `torch.ops._C_ascend.compressor_metadata`。这里的 T 是 RoPE 位置表的行数，H 是 RoPE/head 维度，不是模型全局 hidden size；完整位置表的行数也不是本次调度的 token 数。下表是当时 binding 实际检查的不同对象：

| 对象 | 当时的逻辑约束 | 容易混淆的含义 |
|---|---|---|
| `rope_cos`、`rope_sin` | 非空二维，同 shape、同 dtype | 位置表，不是本轮压缩结果 |
| `cu_seqlens`、`start_pos` | 一维；容量覆盖实际请求 | 请求边界和起始位置，不是物理缓存地址 |
| `kv_block_table` | 二维；行数覆盖实际请求 | 物理 block 映射 |
| `slot_mapping` | flat 或 `(block, offset)` 表示 | 输出行数由压缩后容量决定 |

调用使用 `storage_block_size` 和明确的 slot 格式；原始消费者把返回的 cos/sin 再展平后传给主 Compressor。不能因为两个入口都叫 Compressor，就把主算子的 BSH/TH 输入布局套给 metadata 入口。注册中的 `FORMAT_ND` 描述存储格式，`AutoContiguous` 又涉及适配行为，它们都没有取消二维逻辑 shape 检查。

继续读 C++ binding 还发现，主 Compressor 会读取 `state_cache.stride(0)`，把它作为额外标量传给 `aclnnCompressor`。因此 ABI 对照必须覆盖张量顺序、可选参数、标量顺序及 stride 的含义：只看到同名符号，不能证明另一份库接受同一调用。该次产出是源码接口差分，未完成替换或设备验收；可据此先检查调用双方，再决定需要补哪一种真实 shape/layout 测试。
