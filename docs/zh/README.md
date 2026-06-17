# qwen_cpp 中文文档

qwen_cpp 是一个面向 Transformer 推理路径的 C++ 学习型项目。它当前不定位为可直接部署的大模型推理服务，而是把大模型本地推理系统拆成若干可学习、可测试、可逐步扩展的工程层次：张量容器、基础算子、KV 缓存、prefill/decode 编排、Qwen 风格模型骨架，以及后续需要接入的 tokenizer、runtime、backend、CLI、服务、训练与导出链路。

项目当前最稳定的学习主线是：

```text
tensor -> ops -> cache -> engine -> model
```

这条主线已经纳入 CMake 构建和 CTest 测试。`tokenizer`、`runtime`、`backend`、`cli`、`service`、`python/data_pipeline`、`python/training`、`python/tools` 等目录已经存在，但仍属于后续阶段的脚手架。阅读和开发时应区分“已经可运行的核心模块”和“未来计划中的接口层”。

## 文档导航

| 文档 | 说明 |
| --- | --- |
| [开发者指南](developer-guide.md) | 本地构建、测试、目录职责、贡献流程和常见开发约束。 |
| [模块导读](modules.md) | 按 `tensor/ops/cache/engine/model` 解释当前核心模块职责。 |
| [测试指南](testing.md) | CTest 标签、当前测试覆盖、后续测试分层建议。 |
| [术语表](glossary.md) | Transformer 推理、工程模块和项目内常用术语。 |
| [英文 README](../../README.md) | GitHub 主入口、项目状态、快速开始和路线图摘要。 |
| [架构契约](../ARCHITECTURE.md) | 当前实现的分层边界和依赖规则。 |
| [路线图](../roadmap.md) | 从当前状态到 GGUF 推理、本地训练/微调闭环的阶段计划。 |

## 当前能力边界

当前仓库已经具备可构建、可测试的底层能力。`Tensor` 支持 N 维 row-major 存储、shape/stride 元数据、二维兼容访问、追加、reshape 与边界校验。`ops` 提供 `matmul`、`softmax`、scaled dot-product attention、GQA attention、RoPE 和 causal/additive mask 支持。`cache` 提供多层 KV 缓存、容量限制、只读视图、分配器和缓存管理器。`engine` 已实现 prefill 与单步 decode 的 KV 追加编排。`model` 已经具备最小 Qwen 风格前向骨架，包括 `RMSNorm`、`SelfAttention`、`MLP`、`TransformerBlock` 和 `QwenModel`。

这些能力还没有形成完整本地 LLM 应用。项目当前没有真实 tokenizer 实现，没有 GGUF 加载，没有 prompt 到 token 再到文本生成的闭环，没有可用 CLI 或 HTTP 服务，也没有真实 LoRA/QLoRA 训练、导出和评测流程。文档中凡是涉及这些能力的内容，都应理解为路线规划或接口约束，不能当作已经可运行的功能。

## 推荐学习路径

建议从 `include/tensor.h` 和 `src/tensor.cpp` 开始，理解项目最基础的数据结构。随后阅读 `include/ops` 与 `src/ops` 中的计算算子，重点关注输入 shape 校验、attention mask 规则、GQA head 映射和 RoPE 参数。接着阅读 `include/cache` 与 `src/cache`，理解 KV 缓存如何追加、如何保持层间 token 数一致，以及如何通过视图避免不必要拷贝。读完这些模块后，再进入 `include/engine` 与 `src/engine`，观察 prefill 和 decode 如何把多层 KV 数据写入缓存。最后阅读 `include/model` 与 `src/model`，把算子和缓存能力映射到 Qwen 风格模型结构。

测试文件是当前最准确的可执行文档。每次修改行为前，先查看对应 `tests/*_test.cpp`；每次新增行为后，应补充或更新测试。

## 快速验证

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

按标签运行：

```powershell
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L integration --output-on-failure
ctest --test-dir build -L model --output-on-failure
```

运行 decode 路径基准：

```powershell
.\build\benchmarks\qwen_decode_bench.exe
```

## 文档维护原则

中文文档应保持“当前事实”和“未来规划”分离。已经纳入 CMake、能通过测试验证的能力，可以写成当前能力；仅有头文件、TODO 或未接入构建的模块，应明确标注为脚手架或计划项。新增模块时，应同步更新本目录下的中文说明、根 README 中的文档导航，以及相关测试文档。
