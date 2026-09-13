# 一次 GDN 投影优化审阅发现 FP 单测没有覆盖量化调用路径

2026-09-04 审阅 GDN 投影拆分候选 `91665f45` 时，发现 FP 单测证明的范围小于优化影响的范围。候选在 Qwen3.5 的非 interleaved 分支，将 `in_proj_qkvz(hidden_states)` 后拆输出，改成读取已加载的 weight，切成 QKV、Z 两份各做一次 `F.linear`。这是历史候选的审阅结果；2026-09-12 核对 Ascend main `d4d2957` 时，该分支仍调用投影模块，不能把候选问题当作当前 main 的缺陷。

候选的核心变化可提炼为以下示意逻辑，省略后续 reshape：

```python
# 原路径：由 Linear 模块决定权重表示和计算方式
qkvz, _ = in_proj_qkvz(x)
qkv, z = qkvz.split([qkv_size, z_size], dim=-1)

# 候选路径：假设已加载 weight 仍是普通 [out, in] 浮点矩阵
w = in_proj_qkvz.weight
qkv = F.linear(x, w[:qkv_size])
z = F.linear(x, w[qkv_size:])
```

其中 `qkv_size = (2 * key_dim + value_dim) // tp_size`。普通 FP weight 下，按输出维切权重再投影可以验证拆分计算；候选单测使用这种 Parameter 和 stub，并专门断言模块 `forward` 未调用。问题在于这项断言同时屏蔽了需要验证的量化分发。

源码追踪确认 `create_qkvz_proj` 会把 `quant_config` 传给 `MergedColumnParallelLinear`，其 forward 经 `quant_method.apply` 计算。ModelSlim 的 packed mapping 将 `in_proj_qkvz` 对应到 `in_proj_qkv` 和 `in_proj_z`。以当时 W8A8 static 实现为例，weight 为 INT8，加载后会转置并可能转为 NZ；输入需经 activation quantize，随后 `npu_quant_matmul` 消费 `deq_scale` 及量化修正 bias，输出目标浮点 dtype。裸 `F.linear` 的两次切片没有保留这条路径，不能由 FP 拆分等价推出量化等价。

审阅中还撤回了“遗漏普通 Linear bias”这一疑点，因为该投影构造为 `bias=False`。这与 W8A8 的 `quant_bias` 是两件事：后者参与量化修正，普通层没有 bias 并不意味着可以绕过量化方法。

CI 证据也需要按同一 head 解读：测试选择 job 成功，但执行 NPU 的 jobs 为 `SKIPPED`，`ci-gate` 为 `FAILURE`。因此这里只确认了源码契约缺口和测试覆盖不足，没有观察到该候选的 NPU 数值失败，也没有证明性能收益。复用这次审阅方法时，最有区分力的检查是跟踪加载后 weight 的 dtype/方向/格式以及真正的 quant method，再用真实量化模块验证候选；增加更多普通 FP stub shape 不能补齐这层覆盖。

历史定位：[投影候选代码](https://github.com/vllm-project/vllm-ascend/blob/91665f45b1abccc5651f69e627ae00fc9b049f7f/vllm_ascend/ops/gdn.py#L203)、[W8A8 实现](https://github.com/vllm-project/vllm-ascend/blob/91665f45b1abccc5651f69e627ae00fc9b049f7f/vllm_ascend/quantization/methods/w8a8/w8a8_static.py#L59)、[当次 CI gate](https://github.com/vllm-project/vllm-ascend/actions/runs/33656689331/job/100341791620)。
