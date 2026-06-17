# 模块导读

本文按当前学习主线解释 qwen_cpp 的核心模块。阅读时建议结合源码和测试文件，不要只看头文件接口。

## tensor

`Tensor` 是项目中所有核心计算的基础数据结构。它当前提供 N 维 shape、stride、连续 row-major 存储、二维矩阵兼容访问、span/data 暴露、容量预留、按第 0 维追加、reshape、transpose 和基础统计能力。

关键文件：

```text
include/tensor.h
src/tensor.cpp
tests/tensor_test.cpp
```

需要特别注意空张量契约。默认构造的 `Tensor` 是空张量，`rows()`、`cols()`、`size()` 均为 0。越界访问统一抛出 `std::out_of_range`，空张量调用 `max_value()` 会抛出 `std::runtime_error`。这些行为已经由测试固定，后续修改需要谨慎。

## ops

`ops` 层实现无状态计算算子。当前包括矩阵乘法、softmax 和 attention。

关键文件：

```text
include/ops/matmul.h
include/ops/softmax.h
include/ops/attention.h
src/ops/matmul.cpp
src/ops/softmax.cpp
src/ops/attention.cpp
tests/matmul_test.cpp
tests/softmax_test.cpp
tests/attention_test.cpp
```

attention 是当前最核心的算子。它支持 scaled dot-product attention、additive mask、causal mask、GQA、RoPE 和带 `query_position_offset` 的增量 causal mask。GQA 输入布局固定为 `[seq, heads * head_dim]`，`num_query_heads` 必须能被 `num_kv_heads` 整除。带 KV cache 的增量场景需要正确设置 query 绝对位置，否则 causal mask 会错误屏蔽或放行 token。

`ops` 不应保存状态，也不应知道 session、request、backend 或 HTTP 服务。

## cache

`cache` 层管理推理过程中的 KV 状态。它不做 attention 数学计算，而是负责多层 key/value 张量的存储、追加、容量限制、只读视图和生命周期管理。

关键文件：

```text
include/cache/KVCache.h
include/cache/CacheAllocator.h
include/cache/CacheManager.h
src/cache/KVCache.cpp
src/cache/CacheAllocator.cpp
src/cache/CacheManager.cpp
tests/KVCache_test.cpp
tests/cache_allocator_test.cpp
tests/cache_manager_test.cpp
```

`KVCache::append` 要求 key/value 非空、shape 一致，并且列数等于 `num_heads * head_dim`。追加后的 token 数不能超过 `max_tokens`。`total_token_count()` 表示缓存序列长度，当不同层 token 数不一致时会抛出异常，因为这通常意味着生命周期调用出错。

`CacheAllocator` 提供简单的 Tensor buffer 复用能力。`CacheManager` 负责管理多个 cache id，并为 engine 提供会话级缓存访问入口。

## engine

`engine` 层当前聚焦 prefill 和 decode 的 KV 编排，不是完整文本生成引擎。

关键文件：

```text
include/engine/prefill.h
include/engine/decode.h
src/engine/prefill.cpp
src/engine/decode.cpp
tests/prefill_test.cpp
tests/decode_test.cpp
```

`PrefillEngine` 负责把 prompt 阶段产生的多层 KV 张量批量写入缓存。`DecodeEngine` 负责把当前步单 token 的多层 KV 张量追加到已有缓存。当前实现假设 KV 已由外部提供；也就是说，它验证和编排 cache 写入，但不负责从 token embedding 运行完整模型前向，也不负责 logits 或采样。

后续若要把 `engine` 扩展为真实推理流程，需要明确模型前向、tokenizer、sampling 和 backend 的归属，避免把所有功能堆进同一个类。

## model

`model` 层提供 Qwen 风格最小前向骨架。当前包括配置、权重容器、embedding、RMSNorm、SelfAttention、MLP、TransformerBlock 和 QwenModel。

关键文件：

```text
include/model/*
src/model/*
tests/model_test.cpp
tests/embedding_test.cpp
```

`ModelConfig` 固定了 Qwen/GQA 相关 shape 约束，例如 `hidden_size` 必须能被 `num_attention_heads` 整除，`num_attention_heads` 必须能被 `num_key_value_heads` 整除。`ModelWeights` 当前提供手工填充和 shape 检查路径，还没有真实 manifest/GGUF 权重加载。`QwenModel` 可以对 token embedding 执行最小前向并计算 logits，但这仍不是完整文本生成系统。

## scaffolding modules

以下模块存在文件和接口，但仍是后续阶段脚手架：

| 模块 | 当前状态 |
| --- | --- |
| `tokenizer` | `load/encode/decode` 接口存在，真实分词未实现。 |
| `runtime` | `ModelRunner`、`InferenceSession` 等类型存在，生成流程未实现。 |
| `backend` | `BackendAdapter` 与 `GgufLlamaCppBackend` 存在，真实 GGUF/llama.cpp 接入未实现。 |
| `cli` | 入口和命令函数存在，但未形成可用命令行工具。 |
| `service` | HTTP server 接口存在，但路由和服务逻辑未实现。 |
| `python` | 数据、训练、导出脚本是规划入口，当前不是可运行训练链路。 |

这些模块不应在文档或代码注释中被描述为已完成能力。新增真实实现时，应同步补 CMake、测试和文档。
