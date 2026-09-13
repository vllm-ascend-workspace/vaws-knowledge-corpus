# FlashMLA 缓存的 reader 与切片 writer 需要不同的 stride 证明

2026 年 9 月调查 FlashMLA 适配时，缓存按 token 存放 512 维 NoPE 和 64 维 PE。reader 读取完整 576 维，writer 却分别拿到末维切片。用“首轴非连续”概括两者，会漏掉 writer 看到的行间距与块间距。

以当时的单 KV head、ND 布局为例，下面是帮助理解的地址模型，stride 单位是元素：

```text
完整 cache: [num_blocks, block_size, 1, 576]
NoPE view:  cache[..., :512]
PE view:    cache[..., 512:]
地址 = base + b*s0 + token*s1 + head*s2 + d
```

末维切片的 `stride(-1)` 仍为 1，但 NoPE 行宽只有 512、相邻 token 的实际间距仍为 576；PE 同理。若不同层的页交错在 backing 中，`s0` 还可能大于 `block_size * 576`。因此同一底层缓存可能同时存在第 0 维块间隙和切片后的行间隙。reader 支持完整缓存的首轴 stride，并不能证明 writer 支持这种组合。

当时读取的 `ScatterPaKvCache` 源码分了两层判断。ACLNN 侧的“只有首轴非连续”要求其余轴都连续；tiling 的一般非连续分支则要求 key、keyCache，以及双输入模式下的 value、valueCache **尾轴都连续**，且至少有一处整体不连续。命中后切到非连续模板，并从输入 descriptor 读取多维 stride，而不是用 shape 乘积推导。因此必须确认实际 shape/dtype/layout 进入了这个模板，而不能只确认函数名存在。

把缓存 flatten 为二维也有条件：被合并的相邻轴须满足 `stride[i] == shape[i+1] * stride[i+1]`。存在块间 padding 时，不能把所有 block/token 当成连续行；强行 reshape 若生成副本，可能使后续原地写只更新副本。需要检查 alias/回写语义，而不是只看输出 tensor 数值。

这次只完成源码条件、模板选择和 backing 几何的调查，未验证目标安装包及设备执行。下一次判别应保留 block gap、NoPE/PE 两个切片和未写 guard 区，分别验证 writer 更新原 backing 后再由 reader 读取；这是由调查提出的验收方向，不是当时已经通过的实验。
