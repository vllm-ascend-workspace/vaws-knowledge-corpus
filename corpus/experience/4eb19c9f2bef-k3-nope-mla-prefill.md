# 沿模型计算链纠正对 K3 NoPE 与 MLA prefill 的推断

2026-07 至 08 月，Kimi K3 适配审查中，配置字段和 MLA 名称引出了两个易错推断：看到 `use_rope` 就认为 target 一定做位置旋转；看到压缩 KV cache 就认为 prefill 一定按单头 MQA 计算。调查逐段读取模型构造、MLA wrapper 和 prefill consumer，形成的是源码解释，没有新增 NPU 数值实验。

NoPE 的直接证据在实际构造和传递链：target 使用 `rotary_emb=None`，并向 `MLAModules` 标记 `use_mla_rope=False`。Q/K 中保留位置切片，或者变量名仍含 `_pe`，不等于该路径一定执行了旋转。旧 identity cos/sin 也只是满足接口的占位，不能独立用来判定模型数学语义。draft 的 rotary 构造属于另一 attention group，应单独追踪。

prefill 的源码则明确展开每个 head。以下是省略权重细节后的等价数据流：

```text
q_proj(q_latent) -> view(T, num_heads, qk_head_dim)
                -> q_nope / q_pe
kv_b_proj(k_latent) -> view(T, num_heads, nope_dim + value_dim)
                   -> k_nope / value
k_pe -> 按当前 prefill head 维度展开
```

实际函数还先按 `num_decode_tokens:num_actual_tokens` 取本轮 prefill 子段，再使用对应 slots。由此可见，压缩形式用于缓存存储与 decode 路径，不限制 prefill 必须使用同一种 head 表示；判断算子输入要读投影、view、split 和 expand 之后的真实 tensor。

这次没有通过配置改名或试关开关来“验证”数学定义，也没有证明任意 fused preprocess 都保持 NoPE。早期禁用某项优化只反映当时接口限制，不能推广到后来 A5 或新 wrapper。复用时应核对目标版本从模型字段到实际 consumer 的这条链。[公开模型适配工作](https://github.com/vllm-project/vllm-ascend/pull/14600)。
