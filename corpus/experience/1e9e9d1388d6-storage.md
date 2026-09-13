# 显存归因先排除参数混比，再追查 residual bank 的分配时机

2026 年 7 月 25 日，Kimi K3 的显存调查一度把较高内存占用归因于 `max_model_len`。回看实际启动参数后，早先两组还改变了视觉 profiling 和 `max_num_batched_tokens`，不能从总显存差值直接推断长度参数的作用。

随后用同一五层 INT4 fixture 做受控对照：TP8、eager、`max_num_seqs=4`、`max_num_batched_tokens=8192`、仅语言模型，跳过多模态 profiling，EP 和 FlashComm 均关闭。只改变最大模型长度，worker 的分项记录为：

| max_model_len | peak activation |
| --- | --- |
| 1024 | 约 1.97 GiB |
| 8192 | 约 1.97 GiB |
| 327680 | 约 1.97 GiB |

这组结果否定了“该次 activation 差额由最大长度直接造成”的解释。它没有证明显存与上下文长度普遍无关：实际 batch tokens、KV 分配策略、graph capture、视觉输入等仍会改变内存。当时日志同时分列 weights、peak activation、non-torch memory 和 graph memory；调查使用了分项值，没有把启动后 `npu-smi` 总量当成 activation peak。

8 月 22 日迁移到 v0.27 相关实现时，又遇到另一类 residual storage 问题。旧模型在进入 decoder 层前，按当时全局 token 行数分配 `block_residual[T,B,H]`，B 是残差块容量；后续层经 FlashComm／分片路径得到本地 view。view 的可见 shape 变小，并不会释放其底层全局 storage。因此只打印后续 tensor 的 `numel()`，会漏掉仍由 view 保持存活的分配。

源码定位先找到 `hidden_states.new_empty(T,B,H)` 的分配点，再沿 `maybe_chunk_residual()` 和残差传递检查分配与分片的先后关系。期间有一个改成逐层 `cat` 增长残差集合的对照补丁，用来去掉提前分配整块 bank 的行为；这份对照不是最终推荐实现，也不能只凭代码改动宣称内存收益。

后续实现改为先在模型入口完成 sequence-parallel shard，再按本地 hidden states 行数分配 residual bank。其原因可以用一般布局说明：若 dtype 每元素 b 字节，全局 bank 的逻辑分配为 `T*B*H*b`；让一个本地 view 指向它，不会把分配自动变成 `T_local*B*H*b`，必须改变实际 allocation 的输入 shape。这个推导解释了为什么需要修分配时机，本文不据此公布未做单变量验收的节省量。

两次调查的结果可以同时成立：7 月受控实验排除了一个错误归因；8 月源码则暴露了不同版本中的生命周期问题。前一次“长度三档近似相同”不能拿来否定后一次 storage 回归，后一次修复也不能倒过来解释所有早期显存差值。
