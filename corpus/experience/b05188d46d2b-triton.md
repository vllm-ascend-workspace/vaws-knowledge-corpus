# 图测试通过后继续追查动态长度触发的 Triton 编译

2026 年 9 月 11 日，查询量化和索引整理已经通过数值与图回放测试，后续仍发现在线遇到新 token 数时可能生成新 Triton 变体。原因是每核处理块数随长度改变，却被声明为编译期常量；索引排序的行分块也会变化。捕获过若干图，只证明这些形状可执行，没有证明其他 eager 长度能命中编译缓存。

先枚举实际 launch 参数对应的 JIT 键，区分显式 `constexpr` 与普通参数默认特化。键数只是潜在变体数量，不能当作运行中真实发生的编译次数。修正同时完成两件事，下面是原改动的缩略示意：

```python
# 原参数：BLOCKS_PER_CORE: tl.constexpr
@triton.jit(do_not_specialize=["num_rows", "blocks_per_core"])
def kernel(..., num_rows, blocks_per_core, BLOCK_ROWS: tl.constexpr):
    first_block = tl.program_id(0) * blocks_per_core
    ...  # 根据运行时块数循环
```

仅添加 `do_not_specialize`，但仍让同一个计数作为 `constexpr`，不能消除这条变化来源。查询量化的行块固定；索引整理仍需要静态 `BLOCK_ROWS` 来确定排序 tile，因此选择有限预热，而不是把所有参数都动态化。

预热按排序 scratch 预算求可达的行块上限，再从 1 token 开始，取每个行块跨越核数分界的第一个 token 数，直到配置的最大 batch。例如当时测试中 TopK=2048、40 核、最大 4096 token，只需用 1 和 41 token 覆盖两种可达行块；TopK=128 则需要更多行块代表值。示例用于解释预热选择，不能硬编码到其他核数或 TopK。

验证先执行启动预热，再把 `triton.JITFunction.cache_hook` 换成遇到新编译就抛错的函数。随后遍历 15/16/17、核数前后、两倍核数前后、767/768/769、1023/1024 及 4096 等长度，同时精确比较量化值、scale 和整理后的索引。这比只看运行没有异常更有判别力：任何新增特化都直接使测试失败，数值比较则防止动态循环少处理尾块。

原始联合回归为 149 项通过，包含数值、图回放和预热后变长检查。它证明的是当时配置和选定范围内的特化收敛，没有整网在线尾延迟结果，也不保证新的 dtype、TopK、核数或后续 Triton 版本自动沿用相同缓存行为。
