# 一次用 GDN 业务 metadata 构造 causal conv 阴性复现的记录

2026-06-22 在 A3 上，针对“业务合法输入是否能让 AscendC causal conv 成功而 Triton 越界”的疑问，按 GDN speculative decode 的 metadata 构造输入，再分别调用两种实现。当时版本为 vLLM `967c5c3bc`、Ascend `fc9230c42`；该实验没有复现目标错误。它保留了可复用的输入构造和阴性结果边界，不代表当前 main 的全路径验证。

构造依据是 Qwen3.5 大模型的卷积通道参数：key heads 为 16，value heads 为 64，两者 head dim 均为 128，因此 global dim 为 `2 * 16 * 128 + 64 * 128 = 12288`。卷积 width 为 4、MTP 为 3，state_len 为 `width - 1 + MTP = 6`。固定 `batch=128`、`mamba_cache_mode=none`，使用接近上限的 `context_len=262140` 和 `max_model_len=262144`。

all-max 接受模式每条序列包含 4 个本轮 token，故总输入行为 `128 * 4 = 512`。实际记录的 metadata 是：

- `spec_query_start_loc = [0, 4, 8, ..., 512]`，长度 129；`num_accepted_tokens` 长度 128，全部为 4。
- `spec_state_indices` 形状 `[128, 4]`，开头是 `[[0,1,2,3], [4,5,6,7], ...]`；AscendC 实际收到的 `cache_indices` 是 `[0,4,8,...,508]`，host 参数来源记录为 `spec_decode_fallback_meta`。
- 普通 decode/prefill 数为 0，spec decode 序列数为 128、token 数为 512。

这比只生成相同 shape 的随机张量更有区分力：累计起点、状态槽和接受数需要相互一致，并按各自 wrapper 的业务转换构造调用参数。

| 本地算子构造 | TP8 对应 shape | TP16 对应 shape |
| --- | --- | --- |
| `x`，BF16 | `[512,1536]`，stride `[1536,1]` | `[512,768]`，stride `[768,1]` |
| `conv_state_sd`，BF16 | `[512,6,1536]`，stride `[9216,1536,1]` | `[512,6,768]`，stride `[4608,768,1]` |
| `conv_weights_dw`，BF16 | `[1536,4]` | `[768,4]` |
| 两实现返回 | `[512,1536]`，status ok、finite | `[512,768]`，status ok、finite |

这里 TP 用来决定本地通道维，记录中的算子在单个 NPU 上执行，不能当作 TP8/TP16 分布式服务验收。AscendC 入口为 `npu_causal_conv1d_custom`，Triton 入口为 `causal_conv1d_update_npu`。两边成功并且输出有限，仅排除了这些输入上的目标报错；这不是两实现逐元素数值一致的证明，也没有覆盖其他 cache mode、非连续状态索引、graph capture/replay 或完整服务。

另一个观察是 TP8 对应 shape 首次 Triton 调用的记录耗时约 25.8 秒，后续同 shape 调用约 129 毫秒。这是调用层 elapsed 观察，未拆开编译、初始化与 kernel 时间，不能据此给出稳态性能排名；但它说明首次等待很久仍可能最终正常返回，不能直接把等待判成越界或死锁。原失败若仍未复现，下一步应携带真实场景的 cache mode、状态索引分布和 graph 参数继续缩减，而非把这组阴性输入升级成整体正确性结论。

2026-09-12 查看 Ascend main `d4d2957` 时，GDN spec 分支实际消费的是 custom AscendC operator 与 spec metadata；本次历史双实现比较不应被解读为当前生产分支仍在二选一。
