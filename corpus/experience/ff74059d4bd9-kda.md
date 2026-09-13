# KDA 卷积权重预打包后，还要保持两种 reload 路径和图地址有效

2026 年 7 月 30 日，一次 Kimi KDA 热路径优化把卷积权重的转置、拼接、转 dtype 和连续化移到加载阶段。checkpoint 中 q/k/v 三份权重均为 FP32 `[C,1,W]`，AscendC 消费的是与 activation 同 dtype 的连续 `[W,3*C]`，其中 `C=local_num_heads*head_dim`。每次 forward 重复构造这个张量没有必要，但缓存成一个普通临时属性或 non-persistent buffer 又会脱离已有 reload 接口。

关键约束有两个：完整 checkpoint 重载后，派生 packed 权重必须反映新的 q/k/v；直接按 kernel-format 名称更新时，`get_parameter(name)` 必须找得到实际被 forward 消费的对象。仅“首次加载能跑”无法证明这两条路径成立。图执行还要求已捕获的权重地址保持稳定，重新赋值一个内容正确的新 tensor 仍可能使图引用旧地址。

最终保留 q/k/v 的 canonical 权重，在 `q_conv1d` 上注册不求梯度的命名参数 `packed_conv_weights`。加载后处理包装原有 `process_weights_after_loading()`：先完成原处理，再重建派生布局；遇到 meta 权重时不打包。重建使用 `replace_parameter(..., prefer_copy=True)`，在可复用现有参数时复制数据。forward 的 `_conv_weights_t()` 只返回这个命名参数。

打包顺序可写成如下等价逻辑，Q、K、V 的顺序与沿通道拼接均是算子契约的一部分：

```python
packed = cat([
    q[:, 0, :].T,
    k[:, 0, :].T,
    v[:, 0, :].T,
], dim=1).to(activation_dtype).contiguous()
```

回归使用 `local_num_heads=2、head_dim=3、conv_size=4`，令三个分量取不同的递增数列，验证 `[4,18]` 内容、dtype、contiguous、`named_parameters()` 和 `state_dict()` 中的注册。随后修改一个 canonical 分量并调用加载后处理，断言 packed 内容更新且 `data_ptr()` 不变。

完整 reload 测试实际调用 `record_metadata_for_reloading → initialize_layerwise_reload → weight_loader → finalize_layerwise_reload`，要求返回的 packed 参数与旧对象相同、地址相同、值变为新 q/k/v 的打包结果。另一测试通过完整参数名 `q_conv1d.packed_conv_weights` 直接 `copy_` kernel-format 数据，再确认 forward 返回该对象。这样分别覆盖派生更新和直接参数更新，没有用重新实现的模拟 reload 代替真实生命周期。

当时 main 与 v0.25.1 适配分支各有 31 项 CPU 测试通过记录。这是加载与地址稳定性的验证，没有改动 AscendC kernel，也没有据此宣称全模型服务或图重放重新验收；任意不完整 q/k/v checkpoint 的增量重载仍不在承诺范围内。历史修改可见 [KDA 权重预处理工作](https://github.com/vllm-project/vllm-ascend/pull/12950)。
