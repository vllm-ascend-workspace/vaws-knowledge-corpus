# MRV2 的混合缓存接通后，DSpark 热 prefix 一致性仍未通过

2026 年 8 月 28 日，Kimi K3 在 Ascend 的 MRV2 适配不只是换一个 runner 开关。模型同时使用 KDA／Mamba 状态与 MLA attention cache，Ascend 又将 K/V 绑定成分离的 tensor views。启动问题解决后，真正请求还暴露了状态容量和调度通知缺口；其中若干修复有闭环，但 DSpark 热 prefix 分歧没有解决。

首先对齐 MLA 的 `get_kv_cache_shape(cache_dtype_str=...)` 接口，并让 MRV2 的 `_init_kv_zero_meta()` 在上游实际读取的 `kv_block_zeroer` 属性上设置 `AscendKVBlockZeroer`。分离 KV 不能沿用“每段页大小相同”的假设：MLA 的分量宽度可为 512／64，还可能有空 RoPE 分量。清零实现分别保存每段的 payload 与 page stride，用 block ID 算起始地址，仅清 payload，保留 padding 和同一 allocation 中的其他 view。

对应 NPU 测试把原 buffer 填为 7，构造 BF16 的宽度 512／64、INT8 宽度 32 与空分量；物理 block size 128，逻辑／物理比例覆盖 1 和 3。测试指定清零 `[0]` 或 `[1,3]`，并比较整个 backing buffer，而非只看目标 view，因而能检查相邻块、padding、共享 allocation 中的 Mamba 区域未被越界写零。

随后遇到 `no mamba layers in the model`。只补 Mamba 类型识别可以启动，却没有解决状态列数：上游初始化按裸 `MambaSpec` 添加 speculative state 槽，而 worker 配置保留了 `UniformTypeKVCacheSpecs` 包装。测试使用 4096-token 上限、384-token block 与 7 个 speculative 槽；开启 prefix 时应为 `ceil(4096/384)+7=18` 列，旧路径只给 11 列，1537-token 请求已经需要 12 列。最终在 MRV2 初始化前只展开内容完全相同的 Mamba specs，让类型识别和容量计算一起恢复，保留 attention 分组与调用者配置。

另一条独立缺口是“zeroer 已装上，但新 block IDs 没到”。实际 manager 分配回归期望 `[1,2]`，却拿到 `[]`：当时精确类型白名单没有包含 Ascend MLA／indexer 子类。修复复用 spec registry 和 `FullAttentionManager` 的分配／回收逻辑，仅在既有清零开关开启时转发新 block。一次尝试绕过清零后请求变好，但绕过函数实际调用次数为 0，因此没有被当作“关闭清零修好了数值”的因果证据。

最终含调度修复的单节点 TP16、DP1 无草稿 graph 版本，执行了 17 个推理请求，含 1537-token prefix 冷／热／重置、127／128／129 和 383／384／385／769 等边界、四并发与一次采样。cached tokens 为 `0/1536/0`，三次生成的 32 个 token IDs 相同，16 个 rank 实际调用了新 block 清零。它与此前 eager 的结果一致，但最终版没有重新做完整 eager／graph 双版本矩阵。

同次最终 DSpark eager 结果仍是 cold／reset 一致、warm 在第 8 个 token 分歧，返回 token 也不是那一步所报 top-1；HTTP 200 不足以通过验收。后续 rebase 的 CPU／metadata 测试有通过记录，服务 NPU 矩阵未随 rebase 重跑，不能把更晚测试数与早期设备证据拼成同一版本的完整通过。该经验对应当时的 [MRV2 适配工作](https://github.com/vllm-project/vllm-ascend/pull/15199)，不代表现行版本仍有同样问题。
