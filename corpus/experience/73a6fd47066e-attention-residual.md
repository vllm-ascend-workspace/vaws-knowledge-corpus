# Attention Residual 的数值调查先纠正了 golden 的含义

2026 年 8 月 3 日，Kimi K3 的 AscendC Attention Residual 接入中，融合路径与一份称为“reference”的 NPU 分解实现出现数值差异，随后又观察到 draft 采信变化。最初容易得出的判断是融合算子偏离了模型定义；继续核对上游实际实现后，这个判断被收窄：旧 NPU 分解是一个历史对照，不能直接充当 canonical golden。

输入为 prefix `[T,H]` 和此前有效 residual blocks `[T,N,H]`。把 prefix 加入来源集合后，对每个来源求归一化投影分数，在来源维做 softmax，再对原始 values 加权求和。两种分解在实数算术下等价，浮点运算顺序却不同：旧路径先通过 `npu_rms_norm(values.float(), norm_weight.float(), eps)` 物化带权归一化 values，再矩阵乘 projection；当时上游 NVIDIA／AMD 实际 kernel 先将 norm 与 projection 的权重在 FP32 相乘，点积约简后再乘 reciprocal RMS。

以下仅提炼后者的 score 逻辑，不代表完整 kernel 或执行脚本：

```python
v = values.float()
w = norm_weight.float() * projection_weight.float()
inv_rms = rsqrt(mean(v * v, dim=-1) + eps)
scores = sum(v * w, dim=-1) * inv_rms
probs = softmax(scores, dim=source_axis)
output = weighted_sum(probs, v).to(input_dtype)
```

乘法位置、约简方式与中间 tensor 的物化会改变舍入，不能用“都是 RMSNorm 加线性投影”消去这些差别。原调查进一步核对了上游 CUDA、NVIDIA Triton、AMD Triton 和测试文件的 Git blob，确认所读实现与当时公共 main 对应文件一致。上游测试中的分解表达式和宽松容差也没有声称逐元素 bitwise 相同。

修正后的对照将结果分为 `kernel_vs_canonical`、`kernel_vs_legacy_npu`、`canonical_vs_legacy`，不再把后一组误叫“算子对 reference 的误差”。已有 NPU 测试输入包含 BF16、`H=7168`、`eps=1e-5`，覆盖少量 decode、较多 residual blocks 和 129-token prefill；这能检查算子级误差，却不足以从 tolerance 通过推导 draft 对输入扰动不敏感。

这次真正完成的是参考定义与证据标签的纠正。一个固定长输入、有限输出的服务质量检查，只能说明该请求的表现；不能凭采信变化反推某个 rounding 差异就是唯一根因，更不能把“更接近旧 NPU 路径”自动当作更符合模型。后续复用这类结果时，需要保留三个名字：模型／kernel 的 canonical 运算顺序、历史实现，以及本次待验证的 candidate。
