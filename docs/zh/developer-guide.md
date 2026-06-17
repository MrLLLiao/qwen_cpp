# 开发者指南

本文档面向准备阅读、修改或扩展 qwen_cpp 的开发者。项目当前的核心价值不是提供成品推理服务，而是把 Transformer 推理系统拆解成可学习、可测试的 C++ 工程模块。

## 本地环境

项目使用 CMake 管理构建，要求 CMake 3.20 或更新版本，并需要支持 C++20 的编译器。Windows、Linux 和 macOS 都可以作为开发环境；当前示例命令以 PowerShell 为主。

默认情况下，项目使用 `third_party/simdjson` 中的 vendored 依赖。若要使用系统安装的 simdjson，可在配置时传入：

```powershell
cmake -S . -B build -DQWEN_CPP_USE_BUNDLED_SIMDJSON=OFF
```

基准程序默认参与构建。若只关注库和测试，可以关闭：

```powershell
cmake -S . -B build -DQWEN_CPP_BUILD_BENCHMARKS=OFF
```

## 构建与测试

标准验证流程：

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

按测试标签筛选：

```powershell
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L integration --output-on-failure
ctest --test-dir build -L ops --output-on-failure
ctest --test-dir build -L cache --output-on-failure
ctest --test-dir build -L engine --output-on-failure
ctest --test-dir build -L model --output-on-failure
```

如果修改 attention、KV cache 或 decode 相关路径，建议额外运行 benchmark，观察基本性能趋势是否异常：

```powershell
.\build\benchmarks\qwen_decode_bench.exe
.\build\benchmarks\qwen_decode_bench.exe --prompt 128 --decode 64 --layers 4 --q-heads 8 --kv-heads 2 --head-dim 16
```

## 目录职责

| 路径 | 当前职责 |
| --- | --- |
| `include/` | 对外头文件。当前稳定接口主要在 `tensor`、`ops`、`cache`、`engine`、`model`。 |
| `src/` | C++ 实现。当前 CMake 主要构建 `ops_core`、`cache_core`、`engine_core`、`model_core`。 |
| `tests/` | CTest 测试入口。测试是当前行为契约。 |
| `benchmarks/` | decode 路径微基准，主要覆盖 GQA、RoPE、KVCache append。 |
| `docs/` | 架构、路线图、数据、训练、部署和中文文档。 |
| `models/` | 模型 artifact 目录约定和 manifest 示例，不存放真实模型权重。 |
| `configs/` | 推理、训练、评测配置草稿。当前尚未与 runtime 接通。 |
| `python/` | 未来数据处理、训练和导出脚本。当前多为脚手架。 |
| `third_party/` | 外部依赖源码，默认不在常规功能变更中修改。 |

## 修改代码的基本规则

`ops` 层应保持无状态。新增算子时，优先使用“函数式 API + 配置对象”的形式，输入输出通过 `Tensor` 显式表达。算子可以做 shape 和参数校验，但不能持有跨请求状态，也不能依赖 cache、engine、runtime 或 service。

`cache` 层负责状态和生命周期。KVCache 可以管理容量、追加、视图和 token 计数，但不能决定 prefill/decode 的流程，也不能实现 attention 数学逻辑。

`engine` 层负责流程编排。当前 prefill/decode 的职责是把多层 KV 输入写入指定 cache，并返回追加 token 统计。后续若扩展模型前向、logits 或采样，应先确认这些职责是否属于 engine，避免把底层数学或全局 runtime 生命周期塞进 engine。

`model` 层负责模型结构语义。当前实现的是 Qwen 风格最小前向骨架。新增模型组件时，优先把可复用数学计算放在 `ops`，把权重 shape 和层级语义放在 `model`。

`runtime`、`backend`、`tokenizer`、`cli` 和 `service` 当前仍是脚手架。给这些模块新增真实能力时，应同时接入 CMake、补测试、更新 README 和中文文档，否则容易形成“文件存在但不可用”的维护风险。

## 常见任务建议

修改 `Tensor` 行为时，先看 `tests/tensor_test.cpp`。该文件覆盖空张量契约、N 维访问、异常、append、transpose 和容量相关行为。

修改 attention 时，先看 `tests/attention_test.cpp`，重点确认 additive mask、causal mask、GQA head 映射、RoPE 和非法配置行为。

修改 KV cache 时，先看 `tests/KVCache_test.cpp`、`tests/cache_allocator_test.cpp` 和 `tests/cache_manager_test.cpp`。cache 层的关键风险是 token 计数不一致、容量边界错误和视图越界。

修改 engine 时，先看 `tests/prefill_test.cpp` 和 `tests/decode_test.cpp`。prefill 允许批量追加，decode 当前要求单步 token 追加。

修改 model 时，先看 `tests/model_test.cpp` 与 `tests/embedding_test.cpp`。model 当前只应承诺最小可测前向骨架，不应提前承诺真实 GGUF 权重加载。

## 提交前检查

提交前至少运行完整 CTest。若只改文档，也建议运行 Markdown 链接检查或人工确认相对链接。若改动涉及架构边界、模块成熟度、测试覆盖或路线图，应同步更新 `README.md`、`docs/ARCHITECTURE.md`、`docs/roadmap.md` 或 `docs/zh/` 下的对应中文文档。
