# KdaGateCumsum 的末层错误之前，已有 kernel 清单解析失败

2026 年 7 月 21 日，Kimi KDA 自定义算子在调用 `aclnnKdaGateCumsumGetWorkspaceSize` 时失败。最末层出现 `ADD_TO_LAUNCHER_LIST_AICORE ... failed` 和错误码 161002，容易把调查引向 shape 或算子参数。回读更早的同次底层日志后，发现先发生的是 `ParseDynamicKernelConfig` 失败，随后出现 `Op KdaGateCumsum does not has any binary`。

这个顺序改变了最小排查对象：还没有成功选择可执行的 kernel binary，就不能根据 launcher 的失败直接判断算子数学实现或输入值错误。Python binding 可见、共享库能加载、API 名称存在，也都不能证明 OPP vendor 目录中的 kernel 清单完整且引用文件可读。

当时核对的是运行时真正使用的 vendor 产物，主要包括两层索引：

```text
op_impl/ai_core/tbe/kernel/config/ascend910_93/binary_info_config.json
op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json
```

前者需要找到算子条目及其 `binaryList`，再逐项核对被引用的 binary 与 metadata 文件；后者需要包含对应算子的注册信息。检查不能停在文件中出现了算子字符串，因为字符串计数并不能证明顶层索引结构或引用关系有效。

随后重建、安装 vendor 包，构建记录完成到 `build.sh finished`、installer `SUCCESS`。真实编译路径显示使用 CANN 9.0.1，不能沿用镜像名称推断 toolkit。安装后重新读取两份 JSON：KdaGateCumsum、ChunkKdaFwd、KdaLayoutSwap12 均有顶层条目，binary list 长度分别为 3、7、3；逐条引用的 object 与 JSON 文件存在。这里的数量只描述这次构建，不能变成其他版本必须满足的常数。

这条证据链证明了从“运行时无法解析／找到 binary”到“重建后安装索引与引用文件齐全”的变化。逐项打印文件存在性、大小的输出只证明产物关系，不能当作小形状设备直调有限值；这里没有据此认定模型执行或覆盖这些 binary 的数值矩阵已经通过。

可复用的做法是沿错误发生顺序区分 API 包装层与 kernel 选择阶段，再把构建完成、安装索引完整、实际调用成功、数值正确分别验收。本案例不声称重建适用于所有 161002，也不把后续其他环境的 clean build 或不同芯片交叉编译合并成此次运行通过。
