# 把 SiTU 的计算入口从 CustomOp 构造时机中拆开

2026-08-27，Kimi K3 routed MoE 已进入 SiTU 分支，却在 forward 报 `Current vLLM config is not set`。原始红测栈沿 `SituAndMul.__init__ → dispatch_forward → get_current_vllm_config` 失败：`SituAndMul` 继承 `CustomOp`，构造时就选择平台/编译分派，而真实 worker 的 forward 已离开初始化配置上下文。

有区分力的测试将 reference 模块构造放在合法配置上下文内，然后在上下文外执行实际 MoE helper。旧实现出现三项失败；若把测试整体包进配置上下文，就会掩盖生产入口的错误。

修正 routed 路径时不在热路径临时构造模块，而直接复用与 `SituAndMul.forward_native` 一致的张量计算。其等价计算如下，属于说明性提炼：

```python
gate, up = split_last_dimension_in_half(x)
gate, up = gate.float(), up.float()
gate = beta * tanh(gate / beta) * sigmoid(gate)
if linear_beta is not None:
    up = linear_beta * tanh(up / linear_beta)
y = (gate * up).to(x.dtype)
```

普通 MLP/shared 路径则在模型初始化时创建 SiTU 模块，使构造发生在已有配置上下文中。两条路径的共同要求是保持 gate/up 切分、FP32 中间计算、可选 linear clipping 和最后 dtype 转换，不能为了绕过构造异常顺手改激活公式。

修后同文件记录 42 passed，覆盖多 dtype 和有无 linear clipping。它证明了生命周期与数值表达式对齐，没有实测融合 kernel 的性能、峰值显存或整网精度。后来的公共实现已调整生命周期；这篇留下的是如何让测试真正处在生产调用上下文之外，而不是要求四处新传配置对象。[所在历史集成工作](https://github.com/vllm-project/vllm-ascend/pull/14454)。
