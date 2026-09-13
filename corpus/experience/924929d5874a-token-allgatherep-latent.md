# 从首 token 分叉发现 AllGatherEP 的 latent 归约位置错误

2026-07-24，Kimi K3 的两条 MoE 并行路径在相同验证输入上生成不同首 token。调查先区分 AllGather 的两种语义：沿 token 轴拼接不同请求位置，和将同一 token 的 expert partial 求和。张量 shape 相同，也不能证明各 rank 的同一行代表同一份完整结果。

关键 consumer 是 routed latent 的 output transform，其中包含 RMSNorm 和 up projection。设各 rank 的部分结果为 `z_r`，这里需要先形成完整 latent，再进入非线性 norm：

```text
期望：transform(sum_r z_r)
旧风险：sum_r transform(z_r)
```

这只是原调用链的等价关系，不是说所有操作都必须在通信前或后执行。旧 AllGather 路径把归约留到最终输出，而 transform 已提前消费 partial。最小修改在该条件下把 routed all-reduce 移到 transform 前；shared 部分的归约与最终组合也要配套，避免提前归约后又在最终路径重复相加。

调试在 routed transform 前后、shared 输出和最终合并处保存张量。修前两分支首 token 不同；修后首 token及随后两 token 对齐，仍可见约 0.0009766 的局部差异，rank spread 为零，logprob 也没有宣称完全一致。六项定向测试通过。这些证据支持当时归约位置的修正，而不等于完整模型精度或性能验收。

另一次 g_proj 重复 gather 属于 token 轴重复拼接，不能合成同一根因。SP token shards 的各行可能来自不同 token，不能照此对它们做位置相同的最终 all-reduce。应用这个经历时，应先标注每个中间 tensor 是 token shard、expert partial 还是已归约结果，再决定 norm 前后应保留哪一次通信。
