# 用同一 engine 的 cold—hit—reset 证明 prefix cache 真正复用且结果未变

2026 年 7 月 23 日，一次 Kimi K3 prefix-cache 验证使用五层、真实 INT4 权重派生的小模型。在这个有界 fixture 中，重点是状态复用是否正确，而不是重复请求能否返回 HTTP 200。

测试构造了长度均为 1539 tokens 的请求，共享 1536-token 前缀；当时缓存 block size 为 768，因此共享部分恰好覆盖两个完整 block，并在边界后保留少量继续计算的 tokens。这个构造让“应命中多少”在发送请求之前已经确定，避免用一个太短、未跨缓存块的 prompt 得到零命中后仍宣布通过。

每种执行模式内都使用同一个 engine，顺序执行冷请求、共享前缀请求、显式 reset 后的冷请求。验收同时核对两类信号：缓存观测应为 `0 → 1536 → 0`，数值输出应符合预先指定的比较关系。不能通过重新建 engine 代替 reset，因为那样没有覆盖同一实例清理已有状态的行为。

当时 eager 与 `FULL_DECODE_ONLY` 两次独立运行都观测到前两次 cached tokens 为 `[0,1536]`、reset 返回成功、reset 后 cached tokens 回到 0。冷／命中请求各生成三个 tokens，token IDs 完全相同；被选 token 的 logprob 差值、cumulative logprob 差值均为 0，每步 top-20 的排序相同、数值最大差也为 0。三个不同层次的数值检查有不同作用：最终 token 相等只能排除明显分歧，chosen logprob 能发现同 token 下概率变化，top-20 还能发现候选集合或排序变化。

原始汇总对 reset 后明确给出了缓存归零，但只为 cold／hit 提供了上述逐项 logprob 与 top-20 比较字段。因此不能把这些字段扩大解释为三个阶段所有数值均已比较。运行结束后也确认了本次 NPU 进程退出。

证据范围是这个五层 fixture、单 engine 的缓存生命周期及指定执行模式；它不代表完整 checkpoint 的精度、多 DP 路由一致性，亦没有逐 kernel 证明 graph replay 的每个分支。后续使用这个测试构造时，真正可复用的是“两个完整缓存块加后续 tokens”的输入设计，以及把缓存命中证据和输出一致性证据分开记录。
