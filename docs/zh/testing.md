# 测试指南

qwen_cpp 使用 CTest 管理 C++ 测试。当前测试体系覆盖核心学习主线：`tensor`、`ops`、`cache`、`engine` 和 `model`。真实 tokenizer、runtime、backend、CLI、HTTP 服务和训练链路还没有可用测试。

## 运行全部测试

```powershell
ctest --test-dir build --output-on-failure
```

如果还没有构建：

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## 当前测试标签

| 标签 | 测试目标 | 说明 |
| --- | --- | --- |
| `unit;tensor` | `tensor-test` | Tensor shape、stride、访问、异常、append、transpose。 |
| `unit;ops` | `matmul-test`、`softmax-test`、`attention-test` | 基础算子、mask、GQA、RoPE、非法配置。 |
| `unit;cache` | `kvcache-test`、`cache-allocator-test`、`cache-manager-test` | KV append、容量、视图、生命周期。 |
| `integration;engine;cache` | `prefill-test`、`decode-test` | Engine 通过 CacheManager 写入 KVCache。 |
| `unit;model` | `embedding-test`、`model-test` | Embedding 与 Qwen 风格最小模型组件。 |

按标签运行：

```powershell
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L integration --output-on-failure
ctest --test-dir build -L ops --output-on-failure
ctest --test-dir build -L cache --output-on-failure
ctest --test-dir build -L engine --output-on-failure
ctest --test-dir build -L model --output-on-failure
```

## 测试分层

单元测试应验证单个模块的确定性行为。它们不应依赖真实模型文件、网络、服务进程或 Python 环境。当前 `tensor`、`ops`、`cache` 和 `model` 的多数测试属于这一层。

集成测试应验证稳定模块之间的协作。例如 `prefill-test` 和 `decode-test` 验证 engine 如何通过 `CacheManager` 操作 `KVCache`。集成测试可以跨模块，但仍应保持轻量和可重复。

端到端测试当前尚未接入。只有当 CLI、runtime、backend 和 tokenizer 形成真实可运行链路后，才应增加 e2e 测试。未来 e2e 应覆盖 prompt 输入、模型加载、生成输出、HTTP API 和 artifact 导出加载链路。

## 新增测试建议

新增算子行为时，应同时覆盖正常路径、shape 不匹配、非法配置和边界输入。attention 相关改动尤其需要覆盖 causal mask、additive mask、GQA head 映射、RoPE 参数和增量位置偏移。

新增 cache 行为时，应覆盖空缓存、容量边界、层索引越界、层间 token 数不一致、只读视图和释放/复用行为。

新增 engine 行为时，应明确 prefill 与 decode 的阶段差异。prefill 可以批量追加 token；decode 当前应保持单步 token 追加契约。

新增 model 行为时，应优先测试 shape 约束、权重 ready 状态、最小 forward 输出形状和异常路径。真实权重加载、tokenizer 和 sampling 不应混入当前 model 单元测试，除非对应模块已经有稳定接口。

## Benchmark 与测试的区别

`qwen_decode_bench` 用于观察 decode 路径的性能趋势，不替代功能测试。benchmark 可以帮助发现明显的性能退化，但不应作为判断正确性的唯一依据。行为正确性仍由 CTest 固定。
