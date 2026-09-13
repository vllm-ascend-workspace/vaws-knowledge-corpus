# QuaRot target 接入 MLA draft 时，漏掉了 context_proj 的坐标对齐

2026 年 8 月 10 日，vLLM-Ascend 的 v0.26 系列开发中，QuaRot target 与 MLA DSpark 均能加载，但 draft 的输入对齐仍有缺口。已有逻辑处理名为 `fc` 的 draft 投影；MLA DSpark 在 `combine_hidden_states()` 路径使用的是 `context_proj`，因此同一修正没有执行。

这里必须区分 draft 自身的量化方式与 target 输出的坐标系。即使 draft 使用普通浮点权重，拼接进来的 target 辅助 hidden states 仍可能处于旋转后的基底。问题不在于拼接形状是否匹配，而在于投影矩阵是否与每一块输入使用同一坐标约定。

当时的最小改动保留 `fc` 优先查找，在不存在时查找 `context_proj`。设旋转矩阵是 `R[H,H]`，投影权重为 `W[O,K*H]`，按 target hidden width 将输入维拆成 K 块，分别在 FP32 中右乘 R，最后转回原权重 dtype，并写回原权重。以下是该历史修改的等价逻辑：

```python
assert W.shape[1] % H == 0
blocks = W.float().reshape(O, K, H)
aligned = (blocks @ R.float()).reshape_as(W)
W.copy_(aligned.to(W.dtype))
```

这不是将整个 `K*H` 输入当作一块旋转，也不是给每个辅助层选择不同旋转。历史实现对不能整除 H 的投影打印警告并跳过，避免错误解释布局；它只覆盖既有 `dflash`／`dspark` 方法入口。

新增 CPU 回归使用 `R=[[0,1],[-1,0]]`、`Linear(10,2)`，把权重填为 1 到 20。输入维对应 5 个宽度为 2 的块，期望值直接计算 `initial_weight.reshape(2,5,2) @ R`，再还原为 `[2,10]`。与单位矩阵相比，这个非平凡旋转能暴露漏处理某块、乘法方向错误或只处理 `fc` 的问题。

随后实际 worker 日志显示 `context_proj` 的五块对齐已执行，并记录共享 draft embedding 和 LM head 的复制、坐标对齐路径。共享权重另有测试约束：不要原地改写 target 权重。上述证据证明加载阶段走到了相应修正；不能仅凭日志推导整个推测解码链路的采信率或性能改善，也不构成对其他 QuaRot descriptor、其他辅助层布局的覆盖。

这个案例属于当时的 [MLA DSpark 接入工作](https://github.com/vllm-project/vllm-ascend/pull/13277)。遇到类似问题时，值得先从 `combine_hidden_states()` 实际使用的模块名反向检查加载后处理，而不是只看 draft 配置上是否写了量化。
