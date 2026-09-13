# 用 40→41 task 边界定位 KDA 空 Vnew 分支的事件遗漏

2026-08-07 至 08-10，AscendC `chunk_gated_delta_rule_fwd_h` 的 arch22 路径在多序列短 prefill 挂起。先把整网压缩成单算子，并保留同一旧二进制做相邻边界比较：

| 输入变化 | 观察 |
| --- | --- |
| 40 task | 约 45 ms 完成 |
| 41 task | 超时，退出状态 124 |
| 7 条序列 × 1 token × 6 head，42 task | 触发挂起 |
| 7 条序列 × 2 token × 6 head，仍为 42 task | 能完成 |

task 数相同、只改变序列长度就能改变结果，线索指向空 vector subblock，而不是通信规模。在 Vnew epilogue 中，`rowBegin >= mActual` 的分支不需要计算，但仍占用当前 ping-pong stream 的 workspace event。旧分支只等 `cube1Done`、通知 `vec1Done` 后返回，遗漏 workspace event 的消费与归还；后续轮次复用同一 stream 时便无法继续。

修正保持空分支的计算为空，只补齐协议。下面是原补丁提炼出的事件顺序：

```text
wait cube1Done
if waitWsFromMte3:
    wait MTE3_MTE2[EVENT_ID0 + pingpongFlag]
else:
    wait V_MTE2[EVENT_ID0 + pingpongFlag]
set V_MTE2[EVENT_ID0 + pingpongFlag]
set vec1Done
return
```

关键是依据 workspace 的前一生产者选择等待事件，再统一归还为 V2 所需的状态；不是随意加一次全设备同步。修后原始结果显示 41/42/80/96 task 均完成，42 task 批量执行与七次隔离调用的 h、state、v_new 最大绝对差均为零。

这个闭环只解释当时空 Vnew 分支的形状相关挂起。整网曾同时使用同步兜底，后续仍有 prefix-hit 污染，不能把所有热缓存问题归给这个事件遗漏。参考同类问题时，应核对目标二进制的空分支和事件来源，而非直接照搬旧补丁。[相关历史修复](https://github.com/vllm-project/vllm-ascend/pull/13277)。
