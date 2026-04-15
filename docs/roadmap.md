# 《从当前仓库状态到支持 GGUF 与本地训练/推理的开发路线图》

## 项目现状总结

基于当前仓库的**实际文件结构、源码内容、构建脚本与可执行测试结果**，项目已经不再是“只有算子和 cache 的 PoC”。
它已经进入了**“阶段 0 基线基本收敛完成，阶段 1/2/3/4/6/7/8/10 已建立骨架但尚未闭环”**的状态。

更准确地说，当前仓库包含两类内容：

1. **已经可运行并通过测试的底层核心**
   - `tensor`
   - `ops`
   - `cache`
   - `engine` 中的 `prefill/decode` KV 编排
2. **已经创建目录、头文件、配置和文档，但多数仍为 TODO 骨架的中高层能力**
   - `model`
   - `tokenizer`
   - `runtime`
   - `backend`
   - `cli`
   - `service`
   - `python/data_pipeline`
   - `python/training`
   - `python/tools`
   - `configs/*`
   - `models/manifests/*`

这意味着：

> 当前项目已经具备“**底层推理部件 + KV 编排测试闭环**”，但仍**没有形成可加载 GGUF 模型、可分词、可生成文本、可对外提供服务、可训练/导出的完整产品闭环**。

---

## 当前完成度概括

| 维度 | 现状判断 | 依据 |
|---|---|---|
| 基础张量容器 | 已完成 | `include/tensor.h`、`src/tensor.cpp`、`tests/tensor_test.cpp` |
| 基础算子（matmul / softmax / attention） | 已完成 | `include/ops/*.h`、`src/ops/*.cpp`、对应测试 |
| KV Cache / 分配 / 管理 | 已完成 | `include/cache/*.h`、`src/cache/*.cpp`、对应测试 |
| Prefill 编排 | 已完成（限 KV 写入编排） | `include/engine/prefill.h`、`src/engine/prefill.cpp`、`tests/prefill_test.cpp` |
| Decode 编排 | 已完成（限单步 KV 写入编排） | `include/engine/decode.h`、`src/engine/decode.cpp`、`tests/decode_test.cpp` |
| Model 抽象 | 部分开始，仍以骨架为主 | `include/model/*.h`、`src/model/*.cpp` 已存在，但 `layer/attention` 仍为空壳 |
| LayerNorm 等模型必需算子 | 未开始 / 占位 | `include/ops/layernorm.h` 仍为空，`src/ops/` 无实现 |
| Tokenizer | 已建骨架，未实现 | `include/tokenizer/*`、`src/tokenizer/tokenizer.cpp` |
| Runtime 封装 | 已建骨架，未实现 | `include/runtime/*`、`src/runtime/*` |
| GGUF backend 适配 | 已建骨架，未实现 | `include/backend/*`、`src/backend/gguf_llamacpp_backend.cpp` |
| CLI | 已建入口骨架，未接通 | `src/cli/main.cpp`、`src/cli/commands.cpp` |
| HTTP 服务 | 已建骨架，未接通 | `include/service/*`、`src/service/http_server.cpp` |
| 模型 manifest / 配置 | 已有样例与配置草稿 | `models/manifests/model.manifest.example.json`、`configs/*` |
| 本地训练 / 微调 | 已有 Python 脚手架，未实现 | `python/training/train_lora.py`、`configs/training/lora_sft.yaml` |
| 数据治理 | 已有脚手架与文档，未实现 | `python/data_pipeline/prepare_dataset.py`、`docs/datasets/README.md` |
| 导出链路 | 已有脚手架，未实现 | `python/tools/export_to_gguf.py` |
| 评测与回归体系 | 部分完成 | C++ 单测已覆盖 core/cache/engine；`tests/unit`/`integration`/`e2e` 目录已建立但无真实用例 |
| 文档与部署脚本 | 部分完成 | `README.md`、`docs/ARCHITECTURE.md` 与多个 docs 子目录 README 已存在 |

---

## 可验证现状

已执行并验证：

```powershell docs/roadmap.md
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

验证结果：

- `cmake --build build`：成功
- `ctest --test-dir build --output-on-failure`：**9/9 测试全部通过**

当前通过的测试：

- `tensor-test`
- `matmul-test`
- `softmax-test`
- `attention-test`
- `kvcache-test`
- `cache-allocator-test`
- `cache-manager-test`
- `prefill-test`
- `decode-test`

这说明此前路线图中“`prefill-test` 链接失败、`engine_core` 未链接 `cache_core`”的判断已经**过时**。当前 `CMakeLists.txt` 中：

- `cache_core` 已链接 `ops_core`
- `engine_core` 已链接 `ops_core cache_core`

因此：

> **阶段 0 中最关键的构建闭环问题已经解决。**

不过需要注意：

- 当前 CMake 只构建 `ops_core`、`cache_core`、`engine_core` 与对应测试
- `model/runtime/tokenizer/backend/cli/service` 这些新目录虽然已存在，但**尚未纳入构建闭环**
- 因此“全量构建成功”目前更准确地说是：**已纳入 CMake 的核心库与测试全量成功**，而不是整个仓库的功能模块都已可编译可运行

---

## 已有实现评估

## 1. 已经完成的能力

### 1.1 基础张量与算子能力

以下能力已具备独立可用性，并有测试支撑：

- `Tensor2D`
  - 文件：`include/tensor.h`、`src/tensor.cpp`
  - 能力：构造、访问、转置、填充、最大值读取、边界校验
  - 测试：`tests/tensor_test.cpp`

- `matmul`
  - 文件：`include/ops/matmul.h`、`src/ops/matmul.cpp`
  - 测试：`tests/matmul_test.cpp`

- `softmax`
  - 文件：`include/ops/softmax.h`、`src/ops/softmax.cpp`
  - 测试：`tests/softmax_test.cpp`

- `attention`
  - 文件：`include/ops/attention.h`、`src/ops/attention.cpp`
  - 能力：支持 scaling、manual scale、causal、additive mask
  - 测试：`tests/attention_test.cpp`

**判断**：这是当前仓库最稳定、最可信的部分，已达到“教学/实验型底层模块可复用”的质量水平。

### 1.2 KV Cache 与生命周期管理

以下缓存能力已形成稳定底座：

- `KVCache`
  - 文件：`include/cache/KVCache.h`、`src/cache/KVCache.cpp`
  - 能力：多层 KV 追加、按层查询、token 计数、容量约束

- `CacheAllocator`
  - 文件：`include/cache/CacheAllocator.h`、`src/cache/CacheAllocator.cpp`
  - 能力：按 shape 的简单池化复用

- `CacheManager`
  - 文件：`include/cache/CacheManager.h`、`src/cache/CacheManager.cpp`
  - 能力：多 cache 生命周期管理、容量控制

- 测试：
  - `tests/KVCache_test.cpp`
  - `tests/cache_allocator_test.cpp`
  - `tests/cache_manager_test.cpp`

**判断**：缓存层已经不仅是 PoC，而是一个可继续承载 engine 演进的基础层。

### 1.3 Engine 的 KV 编排能力

当前 `engine` 不再只是 prefill 骨架，`prefill` 与 `decode` 都已有实际实现和测试：

- `include/engine/prefill.h`
- `src/engine/prefill.cpp`
- `tests/prefill_test.cpp`
- `include/engine/decode.h`
- `src/engine/decode.cpp`
- `tests/decode_test.cpp`

当前已实现的能力：

- `PrefillEngine`
  - 校验 layer 输入
  - 校验 token 数一致性
  - 校验 hidden size 与 cache config 一致性
  - 将多层 KV 写入 `KVCache`

- `DecodeEngine`
  - 校验单步 decode 必须是单 token
  - 校验 layer 输入与 cache 状态
  - 将当前步多层 KV 写入 `KVCache`
  - 返回 `appended_tokens` 与 `total_tokens`

**判断**：

- `engine` 层已经完成了“**缓存写入编排**”这一层级的实现
- 但它仍然不是“模型推理引擎”，因为**KV 仍由外部提供，不是模型前向生成**

---

## 2. 已部分实现但未闭环的能力

### 2.1 model / tokenizer / runtime / backend 已经创建骨架，但大多仍为 TODO

相比旧路线图，目前代码仓库已经补入了大量中高层骨架：

#### model

已存在：

- `include/model/model_config.h`
- `include/model/model_weights.h`
- `include/model/transformer_block.h`
- `include/model/layer.h`
- `include/model/attention.h`
- `src/model/model_weights.cpp`
- `src/model/transformer_block.cpp`
- `src/model/attention.cpp`

现状判断：

- `ModelConfig`：已有最小字段骨架
- `ModelWeights`：已有接口但始终返回 `false`
- `TransformerBlock`：已有类与 `forward()` 占位
- `Layer` / `Attention`：仍为空壳
- `src/model/attention.cpp`：空文件

#### tokenizer

已存在：

- `include/tokenizer/tokenizer.h`
- `include/tokenizer/special_tokens.h`
- `src/tokenizer/tokenizer.cpp`

现状判断：

- `Tokenizer::load/encode/decode` 已有接口
- 但当前实现全部为 TODO / 返回空结果或 `false`

#### runtime

已存在：

- `include/runtime/model_runner.h`
- `include/runtime/inference_session.h`
- `include/runtime/generation_config.h`
- `include/runtime/sampling_config.h`
- `src/runtime/model_runner.cpp`
- `src/runtime/inference_session.cpp`

现状判断：

- `InferenceSession` 有最小 `session_id` 封装
- `GenerationConfig` / `SamplingConfig` 有最小字段
- `ModelRunner::load_model/generate` 仍为 TODO，实际不可用

#### backend

已存在：

- `include/backend/backend_adapter.h`
- `include/backend/gguf_llamacpp_backend.h`
- `src/backend/gguf_llamacpp_backend.cpp`

现状判断：

- 已经明确技术方向：GGUF + `llama.cpp` adapter
- 但 `load_model()`、`create_context()` 目前均返回 `false`
- 尚未引入真实 `llama.cpp` 依赖或任何 GGUF 解析/调用逻辑

**结论**：阶段 1 已经不是“未开始”，而是**骨架已落地、实现未闭环**。

### 2.2 CLI / 服务层已开始搭脚手架，但没有产品可用入口

已存在：

- `src/cli/main.cpp`
- `src/cli/commands.cpp`
- `include/cli/commands.h`
- `src/service/http_server.cpp`
- `include/service/http_server.h`
- `include/service/api_error.h`

现状判断：

- CLI 已有单独入口 `src/cli/main.cpp`
- `run_cli()` 当前直接返回 `1`
- HTTP 服务已有 `HttpServer::start/stop` 接口，但实现仍为 TODO
- `ApiErrorCode` 已建立最小错误码枚举
- 根目录 `main.cpp` 仍为空，且当前 CMake 未编译 CLI / service

**结论**：阶段 4 已开始“接口层骨架搭建”，但离“可调用推理服务”仍差完整实现和构建接入。

### 2.3 模型目录、配置与文档骨架已经出现

已存在：

- `models/manifests/model.manifest.example.json`
- `models/gguf/README.md`
- `models/hf/README.md`
- `models/adapters/README.md`
- `configs/inference/default.yaml`
- `configs/training/lora_sft.yaml`
- `configs/evaluation/smoke_eval.yaml`
- `docs/architecture/README.md`
- `docs/datasets/README.md`
- `docs/training/README.md`
- `docs/deployment/README.md`

现状判断：

- 目录规范已经按照路线图初步落地
- 但 manifest 仍是 example，字段为 TODO 值
- 配置文件只有草稿字段，未与代码接通
- 文档多为 TODO 提示，不是可执行说明

**结论**：阶段 2、阶段 10 已经开始，但仍处于“结构先行、内容不足”的状态。

---

## 3. 尚未真正开始或仍几乎为空白的关键能力

### 3.1 GGUF 真实加载与最小推理闭环仍未开始

虽然已经有：

- `backend` 抽象
- `GgufLlamaCppBackend` 类
- 推理配置草稿
- 模型 manifest 样例

但仍缺少任何真实可用能力：

- 无 `llama.cpp` 依赖接入
- 无 GGUF 文件解析与真实加载
- 无 tokenizer 兼容逻辑
- 无 prompt -> token -> prefill -> decode -> text 闭环
- 无 logits / sampling 实现
- 无 smoke inference 测试

**结论**：阶段 3 仅完成“骨架立项”，功能上仍应判定为**未开始/未落地**。

### 3.2 训练、数据治理、导出仍停留在 Python 脚手架

已存在：

- `python/data_pipeline/prepare_dataset.py`
- `python/training/train_lora.py`
- `python/tools/export_to_gguf.py`

但当前状态是：

- 文件只有 docstring + `NotImplementedError`
- 没有真实依赖声明
- 没有数据 schema 实现
- 没有 checkpoint、resume、日志、eval
- 没有 adapter merge 或 GGUF 转换逻辑

**结论**：阶段 6/7/8 都已建文件，但功能上仍应视作**未开始**。

---

## 总体目标与目标架构

## 目标定义

将当前项目从“底层算子 + cache + engine KV 编排的学习型工程”，扩展为**可持续迭代的本地训练与本地推理闭环工程**，同时满足：

1. 支持加载 `GGUF` 模型文件并进行本地推理
2. 支持基于开源训练数据进行本地训练或微调
3. 形成“数据 -> 训练/微调 -> 评测 -> 导出 -> 本地推理”的闭环流程

## 推荐目标架构

建议继续采用**训练与推理解耦、产物通过统一模型清单衔接**的架构；当前仓库目录已经基本按此方向展开，应继续沿此方向收敛，而不是回退到“全部堆在 C++ 工程里”。

### 目标架构分层

```text docs/roadmap.md
docs/
  roadmap.md
  architecture/
  datasets/
  training/
  deployment/

configs/
  inference/
  training/
  evaluation/

models/
  manifests/
  gguf/
  hf/
  adapters/

src/
  core/            # 文档层说明，实际核心实现仍分布于 tensor/ops/cache/engine
  model/           # block / layer / weights abstraction
  tokenizer/       # tokenize / detokenize
  runtime/         # model runner / session / scheduler
  backend/         # gguf backend adapter
  cli/             # 命令行入口
  service/         # 本地 API 服务

python/
  data_pipeline/   # 数据清洗、转换、切分
  training/        # LoRA / QLoRA / eval
  tools/           # artifact convert / validate

tests/
  unit/
  integration/
  e2e/
  *.cpp            # 当前已有核心单测仍位于 tests 根目录
```

---

## 分阶段开发路线图（按当前仓库状态更新）

## 阶段 0：MVP 当前能力梳理与基线收敛

### 阶段目标

把当前仓库整理成**可持续迭代的最小工程基线**。

### 当前状态评估

**基本完成，但尚未彻底收尾。**

已完成：

- `engine_core -> cache_core` 链接问题已修复
- `prefill-test` 已可构建并通过
- `decode-test` 已新增并通过
- 本地标准验证命令可执行：`cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure`
- README 与 `docs/ARCHITECTURE.md` 已能描述底层能力边界

未完成：

- `tests/main_test.cpp` 仍是无关 C 文件，尚未清理或归档
- 根目录 `main.cpp` 仍为空
- 新增模块虽已建目录，但未纳入统一构建与质量检查
- 尚无 CI

### 更新后的完成标志 / 验收标准

- `cmake --build build` 全量成功（针对当前纳入 CMake 的目标）
- `ctest --test-dir build --output-on-failure` 全量通过
- 清理或归档 `tests/main_test.cpp`
- 明确根目录 `main.cpp` 与 `src/cli/main.cpp` 的职责，避免双入口歧义
- 至少补一条自动化校验流程（本地脚本或 CI）

### 下一步建议

阶段 0 剩余工作不应再大范围扩展，只做**收尾和统一**：

1. 清理无关测试文件
2. 统一入口文件策略
3. 把新目录骨架是否纳入构建说清楚
4. 增加 CI

---

## 阶段 1：项目架构整理与模块边界明确

### 阶段目标

把仓库从“底层算子仓库”升级为“具备清晰模型运行边界的推理工程骨架”。

### 当前状态评估

**部分完成（骨架已存在，真实实现未完成）。**

已完成：

- `model`、`tokenizer`、`runtime`、`backend` 目录与头文件已建立
- `GenerationConfig`、`SamplingConfig`、`InferenceSession` 等最小类型已定义
- `BackendAdapter`、`GgufLlamaCppBackend` 已建立抽象方向
- `docs/architecture/README.md` 已预留与 `docs/ARCHITECTURE.md` 对齐的文档位置

未完成：

- `Layer`、`AttentionLayer` 仍为空
- `ModelRunner` 仍不可用
- `Tokenizer` 仍不可用
- backend 抽象过薄，仅有 `load_model/create_context`
- 错误模型尚未统一到 runtime/backend/tokenizer/model 全链路
- 命名空间与模块风格存在割裂：旧 core 代码多为全局风格，新骨架多为 `mini_llm` 命名空间

### 当前阶段的重点任务

- 把阶段 1 从“只有头文件骨架”推进到“**可编译的最小接口层**”
- 优先补齐：
  - `Layer` 最小接口
  - `AttentionLayer` / `TransformerBlock` 的真实前向签名
  - `BackendAdapter` 的 prefill/decode/generate 能力定义
  - runtime 与 tokenizer 的错误返回策略
- 明确新旧模块如何衔接：
  - 旧 `Tensor2D/KVCache/PrefillEngine/DecodeEngine`
  - 新 `mini_llm::{model,runtime,backend}`

---

## 阶段 2：模型管理机制设计

### 阶段目标

建立统一的模型目录、模型元数据、版本管理和运行时配置机制。

### 当前状态评估

**部分完成（目录与样例已存在，机制未实现）。**

已完成：

- `models/manifests/`
- `models/gguf/`
- `models/hf/`
- `models/adapters/`
- `models/manifests/model.manifest.example.json`
- `configs/inference/default.yaml`

未完成：

- manifest 校验器
- manifest 到 runtime/backend 的解析链路
- 模型路径解析与错误提示
- hash 校验、artifact id、license 约束落地
- 单模型 / 多模型切换机制

### 更新后的判断

阶段 2 已不应再标记为“未开始”，更准确应为：

> **目录规范已起步，配置与 manifest 只有静态样例，运行时机制尚未实现。**

---

## 阶段 3：GGUF 文件加载与最小可用推理接入

### 阶段目标

实现**加载 GGUF 模型并完成单轮本地推理**的最小闭环。

### 当前状态评估

**骨架已开始，功能未落地。**

已完成：

- `backend` 目录与 `GgufLlamaCppBackend` 类型已建立
- 配置文件中已出现最小推理参数草稿
- 路线已明确倾向 `llama.cpp`

未完成：

- 真实依赖接入
- GGUF 加载
- tokenizer 对接
- sampling
- 文本生成
- 集成测试

### 当前最短落地路径

1. 在 `backend` 层真正接入 `llama.cpp`
2. 让 `ModelRunner::load_model()` 调用 manifest + backend
3. 实现最小 `Tokenizer` 兼容层或直接委托 backend
4. 打通 `prompt -> generate text`
5. 加一个 smoke test

---

## 阶段 4：推理接口封装（CLI / API / 本地服务）

### 阶段目标

让推理能力可被开发者和本地应用稳定调用。

### 当前状态评估

**已开始搭骨架，但完全不可用。**

已完成：

- `src/cli/main.cpp`
- `run_cli()` 占位接口
- `HttpServer` 占位接口
- `ApiErrorCode` 最小错误码

未完成：

- CLI 子命令解析
- HTTP 路由
- session 管理
- 配置文件启动
- 日志与 streaming
- CMake 接入

### 当前注意事项

当前仓库同时存在：

- 根目录空 `main.cpp`
- `src/cli/main.cpp`

需要尽快统一入口策略，否则后续 CLI/服务接入会持续混乱。

---

## 阶段 5：推理性能优化

### 当前状态评估

**未开始。**

说明：

- 当前只有底层 cache 与 KV 编排测试
- 尚无真实模型推理闭环
- 因此任何性能优化都应在阶段 3/4 打通后再开始

---

## 阶段 6：开源训练数据接入与数据治理

### 当前状态评估

**骨架已开始，功能未落地。**

已完成：

- `python/data_pipeline/prepare_dataset.py`
- `docs/datasets/README.md`

未完成：

- 数据 schema
- 数据下载 / 清洗 / 去重 / 切分逻辑
- license 记录模板落地
- 质量报告产出

---

## 阶段 7：本地训练或微调能力设计与落地

### 当前状态评估

**骨架已开始，功能未落地。**

已完成：

- `python/training/train_lora.py`
- `configs/training/lora_sft.yaml`
- `docs/training/README.md`

未完成：

- 真实训练依赖与环境说明
- LoRA / QLoRA 训练逻辑
- checkpoint / resume
- 训练日志与评测

---

## 阶段 8：训练产物导出与推理侧衔接

### 当前状态评估

**骨架已开始，功能未落地。**

已完成：

- `python/tools/export_to_gguf.py`
- manifest example 可作为未来导出目标格式参考

未完成：

- adapter merge
- HF 导出
- GGUF 转换
- 导出 manifest 生成
- 推理侧回归测试

---

## 阶段 9：模型评测、自动化测试与回归体系

### 当前状态评估

**部分完成。**

已完成：

- C++ 核心单测覆盖 `tensor/ops/cache/engine`
- `tests/unit`、`tests/integration`、`tests/e2e` 目录已建立

未完成：

- tokenizer/runtime/backend/model manifest 测试
- 真实 integration/e2e 用例
- inference smoke test
- Python 侧测试
- 导出 / 训练 smoke test
- CI

### 更新后的判断

阶段 9 的现状应从“只有底层单测”更新为：

> **核心 C++ 单测已形成一条可执行回归线，但还没有覆盖到新引入的上层骨架模块。**

---

## 阶段 10：文档、示例、部署脚本与合规固化

### 当前状态评估

**部分完成。**

已完成：

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/architecture/README.md`
- `docs/datasets/README.md`
- `docs/training/README.md`
- `docs/deployment/README.md`

未完成：

- 可执行的快速开始文档
- CLI/API 示例
- 推理 / 训练 / 导出脚本
- 许可证与安全边界的真实落地文档

### 更新后的判断

阶段 10 已经开始铺文档目录，但内容仍以 TODO 为主，尚未形成交付级文档体系。

---

## 当前所处阶段判断

综合仓库现状，当前项目应判断为：

> **处于“阶段 0 基本完成 + 阶段 1 部分开始 + 多个后续阶段已有骨架但尚未闭环”的状态。**

更具体地说：

- 已完成的是：
  - 底层张量、基础算子、KV cache、cache 生命周期管理
  - `prefill/decode` 的 KV 编排实现与测试
  - 当前纳入 CMake 的核心构建与测试闭环
- 已开始但仍主要是骨架的是：
  - `model`
  - `tokenizer`
  - `runtime`
  - `backend`
  - `cli`
  - `service`
  - `models/manifests`
  - `configs/*`
  - `python/*`
  - `docs/*`
- 还没有真正落地的是：
  - GGUF 加载与文本生成
  - 可用 CLI / API / 本地服务
  - 训练 / 微调
  - 导出 / 评测 / CI / 合规闭环

### 当前阶段结论

当前项目不是：

> “只有底层算子，还没开始往上搭”。

也不是：

> “已经有本地推理 MVP，只差扩展 GGUF 和训练”。

更准确的定位是：

> **“核心底层与 KV 编排 MVP 已经可测可构建；上层推理、GGUF、训练与服务化架构骨架已铺开，但产品级闭环尚未形成。”**

---

## 未完成项清单（按当前仓库重排）

## A. 推理基础闭环仍未完成

- 根目录 `main.cpp` 为空
- `src/cli/main.cpp` 尚未纳入实际可执行构建
- `Tokenizer` 接口存在但无实现
- `ModelRunner` 接口存在但无实现
- `GgufLlamaCppBackend` 接口存在但无实现
- `Layer` / `AttentionLayer` 仍为空壳
- `include/ops/layernorm.h` 为空
- `prefill/decode` 仅完成 KV 写入编排，不是真实模型前向
- 无 logits 输出
- 无采样逻辑
- 无 prompt template 管理

## B. GGUF 支持未落地

- 无真实 GGUF loader
- 无真实 `llama.cpp` 集成
- 无 tokenizer 兼容接入
- 无可运行的 manifest -> backend 加载链路
- 无 GGUF 集成测试

## C. 训练/微调未落地

- 数据脚本、训练脚本、导出脚本均为 `NotImplementedError`
- 无依赖管理
- 无数据 schema
- 无 checkpoint / resume
- 无训练日志与评测

## D. 工程化能力仍不足

- 无 CI
- 无 integration / e2e 实测用例
- 新骨架模块未纳入 CMake 编译
- `tests/main_test.cpp` 仍是无关内容
- 文档多数为 TODO
- 缺少许可证与安全边界落地说明

---

## 下一步优先级建议

## P0：完成“骨架 -> 最小闭环”的第一次跃迁

### 1. 收尾阶段 0

优先完成：

- 清理 `tests/main_test.cpp`
- 明确并统一程序入口（根目录 `main.cpp` vs `src/cli/main.cpp`）
- 为当前仓库增加 CI
- 明确哪些模块已纳入构建，哪些仍仅为脚手架

### 2. 补齐阶段 1 的真正接口层

优先完成：

- `Layer` 最小接口
- `BackendAdapter` 增加真正需要的推理方法
- `ModelRunner`、`Tokenizer` 最小可用实现
- 统一错误模型和命名空间风格

### 3. 把下一里程碑定义为“真实推理 MVP”

建议明确里程碑为：

> **加载 1 个 GGUF 模型，输入 prompt，输出生成文本。**

这一步完成前，不应把当前 `prefill/decode` 当作“推理已完成”。

### 4. 优先接入成熟 GGUF 后端，而不是继续扩展自研 KV 编排层

推荐顺序：

1. 集成 `llama.cpp`
2. 做 backend adapter
3. 接通 `ModelRunner`
4. 提供 CLI
5. 再做本地 HTTP 服务

### 5. Python 训练侧继续保持独立推进

建议按当前已存在目录继续推进：

- `python/data_pipeline`
- `python/training`
- `python/tools`
- `configs/training`
- `models/manifests`

---

## 风险与依赖

## 1. 技术路线风险

### 1.1 当前最大风险不是“没有架构”，而是“骨架过多、落地过少”

仓库已经有较多未来目录和接口，但如果不尽快打通一条真实链路，会出现：

- 目录很多，但都不可运行
- 文档很多，但与实现脱节
- 接口很多，但缺少最小可用实现

因此当前重点应是：

> **减少“只占位”的模块数量，优先打通一条真实路径。**

### 1.2 新旧代码风格割裂

当前仓库存在两套明显风格：

- 旧 core：全局类 / 函数风格
- 新骨架：`mini_llm::*` 命名空间风格

如果不尽快统一，会在阶段 1/3 接入时放大维护成本。

### 1.3 训练与推理格式天然不同

- 训练主格式：HF / safetensors / adapter
- 推理交付格式：GGUF

当前仓库已经开始为此布局 manifest/config/models 目录，这是正确方向，但仍需尽快实现真正的 artifact 管理链路。

---

## 推荐里程碑顺序

建议按以下顺序执行，而不是并行铺开：

1. **完成阶段 0 收尾：清理、统一入口、补 CI**
2. **补齐阶段 1 的最小接口实现，而不是继续只加头文件**
3. **接入 `llama.cpp` 并打通阶段 3 的最小推理闭环**
4. **让 CLI 成为第一个用户入口**
5. **补 manifest/config 与 runtime 的实际对接**
6. **再推进数据治理、LoRA/QLoRA 训练、导出**
7. **最后补全评测、回归、部署、合规**

---

## 最终判断

基于当前仓库实际文件与可验证状态，项目当前最合理的工程定位是：

> **一个已经完成 core/cache/engine-KV 编排验证，并已铺开 model/runtime/backend/cli/service/python 目录骨架，但尚未形成可用本地大模型推理 MVP 的工程。**

因此，要实现“支持 GGUF + 本地训练/微调 + 可持续迭代的训练推理闭环”，最优路线仍然是：

- **短期**：收尾工程基线，统一入口与构建，尽快接入成熟 GGUF 后端，做出可用推理 MVP
- **中期**：接通 manifest/config/runtime/backend/cli，形成可调用本地推理闭环
- **中期后段**：用 Python 训练链路补齐 LoRA / QLoRA、数据治理与导出
- **长期**：通过评测、回归、CI、文档、部署、合规，把训练与推理串成可持续迭代闭环

这条路线与当前仓库的真实完成度最匹配，也最能把已经铺好的骨架转化为真正可交付的能力。
