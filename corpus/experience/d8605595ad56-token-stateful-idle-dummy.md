# 在一个 token 的边界上区分真实 stateful 请求与 idle dummy

2026-08-04 至 08-10，Kimi K3 的 prefix-hit 与 P/D 路径在边界处出现状态分类问题。调查固定 scheduler block 为 384，比较 383/384/385/386 token；关键输入是已经计算 384、当前只调度一个 token 的真实请求。它拥有初始 state，可能选择 decode 图，但 metadata 仍可能被当成普通 prefill，因而图入口、状态读取和实际调度不一致。

不能只用“prompt 是否全部完成”判定该行有无状态。8 月 27 日的后续源码修正进一步检查 `num_computed_tokens > 0`，并结合本次调度宽度、总 token 数判断能否形成 uniform decode；剩余 prompt 也可能短于 speculative width 后被补齐。它并不意味着所有有历史状态的任意 prefill 都能无条件切入同一图档。

第二类一行输入来自 idle DP 的 dummy，含义恰好相反：为了让各 rank 继续协作而执行 dummy，并没有真实请求可推进。旧 dummy 若携带正长度和实际 state slot，会更新不属于本轮请求的状态。修补在 idle 调用传入 `skip_gdn_state_update=True`，使 recurrent metadata 的有效 query 长度为零，并把 state indices 置为 NULL/PAD。

原回归使用四条表面长度均为 8、query 长度均为 0 的行，先把 builder 的 spec/non-spec 控制区填成 77，再检查构造后：actual/decode token 数为零、累计 query offsets 清零、state indices 成为 NULL/PAD。这样能发现沿用上一轮 metadata，不能只检查新分配 buffer 的默认值。

真实 stateful 边界修正后，385-token 请求输出可重复、相关有效状态有限；dummy 修正后一次七条请求质量全部通过，但后来高命中历史 replay 仍只有五条质量通过。因此两个局部修正都不是全部 prefix 污染的根因闭环，后续仍需查 KV 映射和生命周期。[相关历史修复](https://github.com/vllm-project/vllm-ascend/pull/13277)。
