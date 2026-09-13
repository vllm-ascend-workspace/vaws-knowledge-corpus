# 审阅 metadata 独立流时，分开看主机提交、设备依赖与 buffer 复用

2026 年 9 月审阅 [PR #15443](https://github.com/vllm-project/vllm-ascend/pull/15443) 的历史版本时，“异步 metadata”容易被理解成 Python planning、dispatch 和 tiling 都已经搬到后台。源码显示 `DeviceMetadataExecutor.submit()` 仍在调用线程中排序任务、创建或查找事件，并逐个执行 `task.run()`。切换 NPU stream 改变的是设备工作提交到哪里，不会自动产生一个执行 Python 的后台线程。

当时的设备依赖可以提炼为下面的顺序；箭头表示事件先后，不表示 CPU 不再做工作：

```text
主计算流准备本轮输入 → record inputs_ready
metadata 流 wait inputs_ready
metadata 流 wait 上轮 buffer_reusable（若有）
metadata 流按 stage 执行 task.run → record(stage, group_id)
实际消费者 wait 自己的(stage, group_id) → 使用 metadata
消费结束 → 主计算流 record buffer_reusable
下一轮写同一 buffer 前等待这个复用事件
```

这里有两种不同的安全条件：输入就绪事件防止本轮读到尚未完成的输入，复用事件防止下一轮覆盖上一轮仍在读取的持久 buffer。只检查生产者到消费者的等待，会遗漏第二条。原始 `release()` 记录的是当前流上的复用 fence，因此还要沿调用点确认它排在实际消费之后。

FULL 图路径使用按 `(BatchDescriptor, stage, group_id)` 维护的 `ExternalEvent`；相同 batch descriptor 再提交时，代码检查 stage/group 集合没有改变。consumer 等待后执行 event reset，同一轮对已等待的 frontier 去重。这些都是该版本的图与复用约束，不是普通 stream 对象本身提供的保证。

继续看算子 schema，累计长度等可以是 device Tensor，但 `batch_size`、最大长度、head 数、top-k 和布局仍是主机标量。若构造这些标量前需要设备回传，独立流并不会消除它；若值来自有效 CPU mirror，则要确认 mirror 与本轮切片一致。此次工作是源码审阅，未独立量出 TPOT 或尾延迟收益。复用时最有区分力的检查是把 CPU 提交时间、设备 metadata 执行、消费者等待分别观察，而非以“有独立流”替代重叠证据。
