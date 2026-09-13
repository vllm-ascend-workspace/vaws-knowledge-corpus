# 从合法 block ID 追到旧共享 KV backing 的物理地址重叠

2026-08-06，Kimi K3 的旧 GQA DSpark、MLA 与 Mamba 共享 backing 实验路径在长上下文后出现 NaN。逻辑 block table 中没有简单的重复编号，因此调查保留 tensor 的 storage base、storage offset、shape、stride、dtype，以及同一时刻各 group 的 block table，而不是只导出连续 CPU 数值副本。

计算地址时必须区分三层单位：scheduler block、backend 展开后的 page、tensor 实际 stride。下面是通用地址公式，属于这次取证逻辑的提炼：

```text
byte_address = storage_base
             + element_size * (storage_offset + sum(index[i]*stride[i]))
```

现场声明的公共 `page_size_bytes` 为 488,448，但不同 plane 被紧密 view 后，实际行步长并不是这个数字。受害地址减去 draft plane 起点后为 96,780,288 字节，按该 draft 逻辑 block 的实际步长 49,152 字节计算，得到 block 1969；对应 backend page 为 5907，即 1969×3。不能拿公共 page_size 直接反算它。

反算还需与活跃性相互印证：同一阶段 draft block table 的位置 133 包含 1969；该物理地址也对应 target block 5224 展开的 page 15672。前一阶段 target 路径仍有限，下一阶段从历史 KV 读入这段地址时出现 448 个 NaN，而当前新写部分仍有限。于是“两个编号不同”不能排除两段活跃数据实际落在同一物理范围。

这些记录支持旧 plane view 与共享 backing 的地址别名线索。后续隔离 draft backing 仍未消除所有第二次 warmup 污染，所以不能称完整故障已根治，也不能把同时读到坏值等同于捕获了每条设备写指令。旧四平面方案后来从相关 PR 移除，本文的页数和分配方式仅描述历史现场。

这个案例留下的最小取证组合是：受害读取前后的数值、实际地址公式、潜在写入者同阶段的 block table，以及允许共享的生命周期。图内快照也必须由 replay 的设备操作更新；普通 Python forward 打印不能证明每次 replay 时的设备值。
