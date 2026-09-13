# 从异常采样 count=0 追到 KDA 变长临时空间越界

2026-09-08，在 vLLM 0.27.1、Ascend A3、TP16/EP16 的 Kimi K3 缩减专家验证环境中，关闭 prefix cache 后，高并发仍出现连续感叹号和有效采样数为零。排查先区分两种 count=0：未完成的 chunked prefill 被 discard 时可以合法为零；未被 discard 的真实 decode 行为零则需要向前检查。后来捕获到后一种现场，hidden 与 logits 已经全为 NaN，因此不能把采样器的计数当作首个故障点。

保存失败输入后，检查 `chunk_kda_fwd` 的 Post-WU 临时区。host tiling 按实际 packed tokens 分配，设备侧 `WScratchOffset` 却按完整 chunk 行地址化。其差别可用下面的等价容量关系表示，代码是说明性的公式，并非此次执行脚本：

```text
packed_tokens = sum(sequence_lengths)
padded_rows   = sum(ceil(length / chunk_size) * chunk_size)
old_bytes     = packed_tokens * value_heads * key_dim * sizeof(float)
needed_bytes  = padded_rows   * value_heads * key_dim * sizeof(float)
```

原失败输入包含 127 条长短序列、4,720 个 packed tokens，chunk size 为 64。旧区约 13.8 MiB，而 writer 地址范围需要约 23.8 MiB；越过这个区域后会覆盖紧随其后的 workspace。序列多但各自很短时，不能用“总 token 看起来不大”排除这种越界。

最小修改落在 `Tiling4ChunkKdaFwd`：在非融合 Post-WU 分支，把 `tokenHeads * kDim` 改为 `hChunkCount * vHeads * chunkSize * kDim`，让 cursor 与后续区域的起点一起后移，没有用输出补零遮住症状。保存的失败输入回放恢复有限值；8 项 NPU 回归覆盖变长 127/128 条、固定长度 80/129 及两种 recompute 模式。服务三个阶段共 896/896 请求成功，16 rank 的有效 count 在 1–8。

服务对照两边都带有独立的 sampler sort 显存修复，所以保存输入的单算子回放是区分两个问题的重要证据。缩减权重不证明完整模型精度，这些结果也不宣称当前 main 仍有同一缺陷。[当时针对 release 分支的修复](https://github.com/vllm-project/vllm-ascend/pull/16087)。
