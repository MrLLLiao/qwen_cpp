# 《从 MVP 到支持 GGUF 与本地训练/推理的开发路线图》

## 项目现状总结

当前仓库更准确地说处于**“算子与缓存能力验证阶段”**，还没有达到“可对外使用的本地推理 MVP”。  
从实际文件和可验证结果看，项目已经完成了部分 Transformer 推理底座，但**尚未形成可加载模型、可分词、可生成文本、可对外提供接口、可训练/微调、可评测回归的完整闭环**。

### 当前完成度概括

| 维度 | 现状判断 | 依据 |
|---|---|---|
| 基础张量容器 | 已完成 | `include/tensor.h`、`src/tensor.cpp`、`tests/tensor_test.cpp` |
| 基础算子（matmul/softmax/attention） | 已完成 | `include/ops/*.h`、`src/ops/*.cpp`、对应测试 |
| KV Cache / 分配 / 管理 | 已完成 | `include/cache/*.h`、`src/cache/*.cpp`、对应测试 |
| Prefill 编排 | 部分完成 | `include/engine/prefill.h`、`src/engine/prefill.cpp`、`tests/prefill_test.cpp` 已存在，但当前构建链未打通 |
| Decode 编排 | 未开始 / 仅接口 | `include/engine/decode.h` 已定义，`src/engine/decode.cpp` 为空 |
| Model 抽象 | 未开始 / 占位 | `include/model/layer.h`、`include/model/attention.h`、`src/model/attention.cpp` 基本为空 |
| LayerNorm 等模型必需算子 | 未开始 / 占位 | `include/ops/layernorm.h` 为空，`src/ops/` 无实现 |
| Tokenizer | 未开始 | `src/tokenizer/` 空目录 |
| Runtime 封装 | 未开始 | `src/runtime/` 空目录 |
| CLI / API / 本地服务 | 未开始 | `main.cpp` 为空，仓库无服务端或 CLI 接口实现 |
| GGUF 模型加载 | 未开始 | 仓库中无 GGUF 解析、模型权重加载、后端集成相关文件 |
| 本地训练 / 微调 | 未开始 | 无训练脚本、无训练依赖、无数据处理、无 checkpoint 管理 |
| 评测与回归体系 | 未开始 / 极弱 | 仅底层单元测试，缺少端到端推理与训练评测 |
| 文档与部署脚本 | 部分完成 | `README.md`、`docs/ARCHITECTURE.md` 存在，但无运行闭环文档与脚本 |

### 可验证现状

已执行并验证：

- 通过的测试：`tensor-test`、`matmul-test`、`softmax-test`、`attention-test`、`kvcache-test`、`cache-allocator-test`、`cache-manager-test`
- 当前全量构建/测试**不能完全通过**：`prefill-test` 链接失败

失败原因可从仓库结构直接判断：

- `CMakeLists.txt` 中 `engine_core` 仅链接了 `ops_core`
- 但 `src/engine/prefill.cpp` 实际依赖 `CacheManager` / `KVCache`
- 因此 `prefill-test` 在链接阶段出现 `CacheManager`、`KVCache` 未解析符号

这说明项目虽然开始进入 engine 层，但**工程集成仍未闭环**。

---

## 已有实现评估

## 1. 已经完成的能力

### 1.1 基础张量与算子能力

以下能力已具备独立可用性，并且有测试支撑：

- `Tensor2D` 二维张量容器  
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

**判断**：这部分属于“底层数学与容器能力”阶段，质量好于仓库其他部分，是当前最稳的基础。

### 1.2 KV Cache 与生命周期管理

以下缓存能力已形成较完整的底座：

- `KVCache`
  - 文件：`include/cache/KVCache.h`、`src/cache/KVCache.cpp`
  - 能力：多层 KV 追加、行视图、token 计数、容量约束

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

**判断**：缓存层已具备后续承载 prefill/decode 的基础，但当前实现仍偏学习型/单机场景，距离高性能推理 runtime 还有明显差距。

---

## 2. 已部分实现但未闭环的能力

### 2.1 Engine 的 prefill 雏形已存在，但工程未打通

- 接口与实现：
  - `include/engine/prefill.h`
  - `src/engine/prefill.cpp`
- 测试：
  - `tests/prefill_test.cpp`

已有能力：

- 能描述 `PrefillRequest` / `PrefillLayerKV` / `PrefillResult`
- 能对 layer 输入进行基础校验
- 能调用 `KVCache::append` 完成多层 KV 写入

未闭环点：

- `CMakeLists.txt` 中 `engine_core` 未链接 `cache_core`，导致 `prefill-test` 无法链接通过
- `PrefillEngine` 仅处理“外部已给定 KV 的写入”，**不是模型前向生成 KV**
- 没有与 tokenizer、权重、模型层、采样、输出 token 建立连接
- 缺少端到端“prompt -> logits -> next token”的能力

**结论**：prefill 目前只是“缓存写入编排”，还不是“真实推理 prefill”。

### 2.2 架构文档已存在，但代码边界未真正落地

- 文档：
  - `README.md`
  - `docs/ARCHITECTURE.md`

文档已经明确了：

- `ops`：纯计算
- `cache`：状态与生命周期
- `engine`：流程编排
- `model`：层级抽象

但代码层面仍未完成对应落地：

- `include/model/layer.h` 为空
- `include/model/attention.h` 为空
- `src/model/attention.cpp` 为空
- `src/runtime/` 空目录
- `src/tokenizer/` 空目录

**结论**：架构边界有设计意图，但核心中层抽象尚未实现。

---

## 3. 尚未开始的关键能力

### 3.1 GGUF 相关能力完全缺失

仓库中没有看到以下任一项：

- GGUF 文件解析
- 模型元信息读取
- 权重映射与张量加载
- tokenizer 读取或兼容层
- 推理后端选择与封装
- GGUF 兼容性测试

**结论**：GGUF 支持当前应判定为**未开始**。

### 3.2 本地训练/微调能力完全缺失

仓库中没有看到以下内容：

- 训练框架依赖
- 数据集加载脚本
- 数据格式转换脚本
- LoRA / QLoRA / 全量训练实现
- checkpoint、resume、日志、评测
- 导出推理可消费产物的流程

**结论**：本地训练/微调当前应判定为**未开始**。

### 3.3 对外使用层完全缺失

仓库中没有形成真正可用入口：

- `main.cpp` 为空
- 无 CLI
- 无 API 服务
- 无本地 daemon / gRPC / HTTP
- 无配置文件
- 无模型目录规范
- 无脚本化启动方式

**结论**：当前不能作为“本地模型推理产品”使用。

---

## 总体目标与目标架构

## 目标定义

将当前项目从“底层算子 + cache 的学习型工程”，扩展为**可持续迭代的本地训练与本地推理闭环工程**，同时满足：

1. 支持加载 `GGUF` 模型文件并进行本地推理  
2. 支持基于开源训练数据进行本地训练或微调  
3. 形成“数据 -> 训练/微调 -> 评测 -> 导出 -> 本地推理”的闭环流程  

## 推荐目标架构

建议采用**训练与推理解耦、产物通过统一模型清单衔接**的架构，而不是强行在当前 C++ 工程里同时实现训练和推理内核。

### 目标架构分层

```text
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
  core/            # tensor / ops / cache
  model/           # block / layer / weights abstraction
  tokenizer/       # tokenize / detokenize
  runtime/         # model runner / session / scheduler
  backend/         # gguf backend adapter
  cli/             # 命令行入口
  service/         # 本地 API 服务

python/
  data_pipeline/   # 数据清洗、转换、切分
  training/        # LoRA / QLoRA / eval / export
  tools/           # artifact convert / validate

tests/
  unit/
  integration/
  e2e/
```

## 推荐技术路线

### 推理侧推荐方案

**推荐优先集成现成 GGUF 推理后端，而不是自研 GGUF runtime。**

优先建议：

- **方案 A（推荐）**：集成 `llama.cpp` 作为 GGUF 推理后端
- 通过本项目的 `backend` 层封装模型加载、上下文创建、推理参数、tokenizer、采样、session cache
- 本项目保留现有 `ops/cache/engine` 作为学习和实验基础，但产品化推理优先走成熟后端

推荐理由：

- 当前仓库离“完整模型前向 + tokenizer + sampling + weight loader”差距很大
- GGUF 生态成熟实现主要集中在 `llama.cpp` 路线
- 自研 GGUF 解析和全推理 runtime 会显著拖慢交付

### 训练侧推荐方案

**推荐训练用 Python / PyTorch 体系，推理用 C++ 本地 runtime。**

优先建议：

- `Transformers + PEFT + bitsandbytes + datasets + accelerate`
- 训练主路径以 **LoRA / QLoRA** 为主
- 全量训练仅作为远期可选，不应在当前阶段作为主线

推荐理由：

- GGUF 是推理分发格式，不是训练主格式
- 当前 C++ 项目不具备训练图、优化器、反向传播基础
- 用 Python 训练栈可快速打通数据、训练、评测、导出

### 产物衔接原则

训练产物与推理产物不要混用格式：

- 训练中间产物：`safetensors` / adapter 权重 / checkpoint
- 推理交付产物：`GGUF`
- 通过统一 `manifest` 描述：
  - 基座模型
  - adapter 来源
  - 合并方式
  - 导出脚本版本
  - tokenizer 版本
  - 许可证与安全说明

---

## 分阶段开发路线图

## 阶段 0：MVP 当前能力梳理与基线收敛

### 阶段目标

把当前仓库从“部分代码可运行、部分代码只是占位”的状态，整理成**可持续迭代的最小工程基线**。

### 需要完成的具体任务清单

- 修复 `engine_core` 的链接依赖问题
  - `CMakeLists.txt` 中让 `engine_core` 正确链接 `cache_core`
- 让 `prefill-test` 真正构建并运行
- 清理明显无关或误导性内容
  - 如 `tests/main_test.cpp` 为独立 C 代码，且未纳入当前 C++ 测试体系，应移出或单独归档
- 明确空目录与占位模块的状态
  - `src/runtime/`
  - `src/tokenizer/`
  - `include/model/*`
  - `src/model/*`
- 统一目录命名和模块命名规范
- 补一份“当前仓库能力边界说明”
- 增加基础 CI 或至少本地一键验证命令

### 关键技术点

- CMake target 依赖梳理
- 测试入口规范化
- 占位代码与真实能力的边界说明
- 最小可持续交付基线

### 依赖关系

- 无前置依赖
- 是后续所有阶段的基础

### 完成标志 / 验收标准

- `cmake --build build` 全量成功
- `ctest --test-dir build --output-on-failure` 全量通过
- README 能明确说明当前可运行能力与不可用能力
- 测试目录不再混入无关文件
- 空模块有 TODO/状态说明，不再造成误解

### 对应当前项目状态评估

**部分完成**

- 已有基础构建与多数单测
- 但全量构建未通过，说明阶段尚未完成

### 风险点与注意事项

- 如果不先做基线收敛，后续引入 GGUF 和训练模块时会放大工程债务
- 当前 `engine` 层的“已实现”容易被误判为“已具备推理能力”，需要文档澄清

---

## 阶段 1：项目架构整理与模块边界明确

### 阶段目标

把仓库从“底层算子仓库”升级为“具备清晰模型运行边界的推理工程骨架”。

### 需要完成的具体任务清单

- 补齐 `model` 层最小抽象
  - `Layer`
  - `AttentionLayer`
  - `TransformerBlock`
  - `ModelWeights` / `ModelConfig`
- 定义 `runtime` 层接口
  - `ModelRunner`
  - `InferenceSession`
  - `GenerationConfig`
  - `SamplingConfig`
- 定义 `tokenizer` 层接口
  - `encode`
  - `decode`
  - special tokens 管理
- 定义 `backend` 抽象
  - `load_model`
  - `create_context`
  - `infer_prefill`
  - `infer_decode`
- 定义统一错误模型
  - 参数错误
  - 模型加载错误
  - tokenizer 错误
  - OOM / capacity 错误
- 定义统一配置结构与模型清单结构

### 关键技术点

- 抽象边界稳定性优先
- 不把 GGUF 细节泄露到上层业务接口
- 不把训练框架耦合进 C++ runtime
- session 与 model 生命周期解耦

### 依赖关系

- 依赖阶段 0 完成
- 是 GGUF 接入、CLI/API、评测的前置阶段

### 完成标志 / 验收标准

- `include/model/` 不再为空壳
- `src/runtime/`、`src/tokenizer/` 至少有接口骨架与最小实现
- 形成模块依赖图并与 `docs/ARCHITECTURE.md` 保持一致
- 代码中不存在“engine 直接替代 model/runtime”式越层实现

### 对应当前项目状态评估

**未开始**

- 文档有边界定义
- 代码尚未形成真正抽象层

### 风险点与注意事项

- 过早写死模型细节会导致未来只支持单一模型家族
- 抽象过度也会拖慢落地，建议先按“GGUF 推理最小需要”设计

---

## 阶段 2：模型管理机制设计

### 阶段目标

建立统一的模型目录、模型元数据、版本管理和运行时配置机制，为 GGUF 推理和训练产物衔接提供基础。

### 需要完成的具体任务清单

- 设计模型目录规范，例如：
  - `models/manifests/*.json`
  - `models/gguf/...`
  - `models/hf/...`
  - `models/adapters/...`
- 定义模型清单字段
  - 模型名称
  - 版本
  - 模型家族
  - tokenizer 来源
  - 上下文长度
  - 量化信息
  - license
  - 文件校验和
  - 导出来源
- 增加模型注册与校验工具
- 增加模型路径解析与错误提示
- 支持单模型 / 多模型切换配置
- 为推理和训练产物建立统一 artifact id

### 关键技术点

- manifest 版本化
- 文件存在性与 hash 校验
- 模型配置与运行参数分离
- 许可证元信息绑定到模型产物

### 依赖关系

- 依赖阶段 1 的 runtime / backend 抽象
- 是 GGUF、训练产物导出、部署脚本的前置条件

### 完成标志 / 验收标准

- 给定一个模型目录，可以通过 manifest 校验并成功初始化 runtime
- 模型切换不依赖硬编码路径
- 运行日志中可打印模型版本、量化类型、上下文长度、license

### 对应当前项目状态评估

**未开始**

- 仓库中没有配置文件、manifest、模型目录规范

### 风险点与注意事项

- 如果不先设计 artifact 规范，后续训练产物与推理产物会混乱
- 许可证信息必须随模型一起管理，不能后补

---

## 阶段 3：GGUF 文件加载与最小可用推理接入

### 阶段目标

实现**加载 GGUF 模型并完成单轮本地推理**的最小闭环。

### 需要完成的具体任务清单

- 选定 GGUF 推理后端
  - 推荐：`llama.cpp`
- 增加 backend adapter
  - 模型加载
  - context 创建
  - tokenizer 调用
  - prompt prefill
  - decode loop
  - logits 获取
- 设计推理参数
  - `n_ctx`
  - `n_threads`
  - `n_gpu_layers`
  - `batch_size`
  - `temperature`
  - `top_k`
  - `top_p`
  - `repeat_penalty`
  - `seed`
- 增加模型加载错误处理
  - 文件不存在
  - GGUF 版本不兼容
  - tokenizer 缺失
  - 上下文长度不足
- 增加最小生成流程
  - prompt -> tokens -> prefill -> decode -> text
- 增加 smoke test
  - 固定 prompt 返回非空生成
- 明确当前自研 `ops/cache/engine` 与外部 backend 的关系
  - 学习/实验能力保留
  - 产品可用推理以 backend 为主

### 关键技术点

- GGUF 元信息读取
- tokenizer 兼容
- session/context 生命周期
- prompt cache 与 decode step
- CPU / GPU 运行参数适配
- 错误和日志可观测性

### 依赖关系

- 依赖阶段 1、阶段 2
- 是 CLI/API、本地服务的前置阶段

### 完成标志 / 验收标准

- 命令行可加载指定 GGUF 模型
- 能完成单轮对话生成并输出文本
- 支持基础推理参数配置
- 对无效模型路径、无效参数、上下文超限有明确报错
- 至少有 1 个集成测试验证加载与生成流程

### 对应当前项目状态评估

**未开始**

- 仓库中没有任何 GGUF、tokenizer、采样、文本生成代码

### 风险点与注意事项

- 自研 GGUF parser/runtime 风险高，不建议作为当前主线
- 不同 GGUF 模型家族对 tokenizer 和 prompt template 有差异，需要 manifest 管理
- 需要明确 CPU-only 与 GPU-offload 的兼容策略

---

## 阶段 4：推理接口封装（CLI / API / 本地服务）

### 阶段目标

让推理能力可被开发者和本地应用稳定调用，而不是停留在库级别。

### 需要完成的具体任务清单

- 实现 CLI
  - `run`
  - `chat`
  - `bench`
  - `inspect-model`
- 实现本地服务
  - 推荐先做 HTTP JSON API
  - 最小接口：
    - `/health`
    - `/models`
    - `/generate`
    - `/chat`
- 支持流式输出
- 支持 session 管理
- 支持配置文件启动
- 支持日志级别与运行日志输出
- 增加示例请求与启动脚本

### 关键技术点

- 请求参数校验
- session 与上下文复用
- streaming token 输出
- 超时、中断、并发限制
- 错误码设计

### 依赖关系

- 依赖阶段 3 的可用推理闭环
- 与阶段 5 性能优化并行推进

### 完成标志 / 验收标准

- CLI 能完成模型加载与生成
- HTTP API 能返回一次完整推理结果
- 流式接口可逐 token 输出
- 并发超限或模型未加载时返回明确错误
- 提供最小开发者使用文档

### 对应当前项目状态评估

**未开始**

- `main.cpp` 为空
- 仓库中无 CLI/API/服务实现

### 风险点与注意事项

- 先做“单模型、单进程、本地服务”即可，不建议一开始做复杂调度
- session 管理和模型实例管理不能耦合到 HTTP 层

---

## 阶段 5：推理性能优化

### 阶段目标

在具备可用推理闭环后，再系统优化吞吐、时延、内存占用和并发能力。

### 需要完成的具体任务清单

- 增加性能基线测试
  - TTFT
  - tokens/s
  - 峰值内存
  - prompt cache 命中率
- 优化上下文管理
  - KV cache 复用
  - session 清理策略
  - 长上下文截断策略
- 优化并发
  - 单模型多会话
  - 请求排队
  - worker 池
- 优化参数配置
  - batch size
  - threads
  - gpu layers
- 增加缓存策略
  - prompt 前缀缓存
  - 编码缓存
- 增加资源保护
  - OOM 保护
  - 最大上下文 / 最大输出 token 限制
- 增加基准测试报告与回归阈值

### 关键技术点

- KV cache 生命周期
- 前缀缓存设计
- 线程模型
- 内存池和上下文碎片控制
- 量化模型适配
- profiling 与回归基线

### 依赖关系

- 依赖阶段 3、4
- 与阶段 9 的测试体系联动

### 完成标志 / 验收标准

- 有稳定的基准脚本和基准结果存档
- 支持至少一种可复用的 prompt cache 方案
- 并发场景下无明显资源泄漏和上下文串扰
- 对典型硬件有推荐配置

### 对应当前项目状态评估

**未开始**

- 当前仅有底层 cache，不等于可用的推理性能体系

### 风险点与注意事项

- 不要在没有可用推理闭环前过早优化底层算子
- 当前自研 `KVCache` 与成熟后端 cache 可能重复，需明确定位

---

## 阶段 6：开源训练数据接入与数据治理

### 阶段目标

建立可持续的数据处理链路，为本地微调提供干净、可追溯、可复现的数据输入。

### 需要完成的具体任务清单

- 选定首批开源数据源
  - 指令数据
  - 对话数据
  - 领域数据
- 为每类数据记录：
  - 来源
  - 下载方式
  - 许可证
  - 使用限制
- 建立原始数据存储规范
- 建立数据清洗脚本
  - 去重
  - 空样本过滤
  - 编码异常处理
  - 长度过滤
  - 敏感内容规则过滤
- 建立格式转换
  - 转为统一 SFT JSONL 格式
- 定义标注规范
  - `instruction`
  - `input`
  - `output`
  - `system`
  - `metadata`
- 建立数据切分
  - train / valid / test
- 建立数据版本管理
  - 数据集版本号
  - 清洗脚本版本
  - 样本数量统计
- 输出数据质量报告

### 关键技术点

- 数据 schema 标准化
- 清洗与过滤规则版本化
- 数据许可证记录
- 隐私与敏感信息控制
- 可复现的数据快照

### 依赖关系

- 可与阶段 3、4 并行
- 是阶段 7 训练能力的前置条件

### 完成标志 / 验收标准

- 至少有一套可复现的数据处理脚本
- 可稳定产出训练可用 JSONL
- 每份数据集有 license 与来源记录
- 训练前可以自动做数据质量检查

### 对应当前项目状态评估

**未开始**

- 仓库中无数据处理脚本、无 dataset 目录、无数据 schema

### 风险点与注意事项

- 数据许可证与用途限制必须前置，不可在训练完成后补
- 对话模板与 tokenizer 需与目标模型保持一致

---

## 阶段 7：本地训练或微调能力设计与落地

### 阶段目标

形成可在本地或单机工作站执行的微调能力，优先支持 **LoRA / QLoRA**。

### 需要完成的具体任务清单

- 确定训练栈
  - 推荐：`Transformers + PEFT + Accelerate + bitsandbytes + datasets`
- 增加训练配置文件
  - 基座模型
  - 数据集版本
  - batch size
  - grad accumulation
  - lr
  - epochs
  - lora rank/alpha/dropout
  - max length
  - precision
- 增加训练脚本
  - SFT LoRA
  - QLoRA
- 增加日志与追踪
  - loss
  - lr
  - samples/sec
  - GPU memory
- 增加 checkpoint 管理
  - step 保存
  - best 保存
  - resume
- 增加异常恢复
  - 断点续训
  - 配置校验
- 增加硬件说明
  - CPU-only 不推荐训练
  - 最低 GPU/内存建议
- 明确全量训练策略
  - 当前阶段建议仅文档说明，不作为首批交付

### 关键技术点

- LoRA / QLoRA 参数配置
- 显存控制
- gradient accumulation
- resume 机制
- 可重复训练配置
- tokenizer 与 prompt template 一致性

### 依赖关系

- 依赖阶段 6 数据治理
- 与阶段 8 导出衔接强相关

### 完成标志 / 验收标准

- 给定配置文件与数据集，可启动一次 LoRA/QLoRA 训练
- 训练过程有日志、checkpoint、resume
- 训练结束可产出 adapter 权重与训练报告
- 至少有 1 条 smoke pipeline：小数据集训练 1~2 step 成功

### 对应当前项目状态评估

**未开始**

- 仓库中无训练框架、无配置、无脚本、无 checkpoint 机制

### 风险点与注意事项

- 不建议在当前仓库中用 C++ 自研训练图与优化器
- QLoRA 对环境依赖较强，需要明确 CUDA / 驱动 / 库版本矩阵
- 训练配置必须版本化，否则无法复现实验

---

## 阶段 8：训练产物导出与推理侧衔接

### 阶段目标

将训练输出稳定转换为推理可消费产物，并让本地 GGUF 推理链路可直接使用微调后的模型。

### 需要完成的具体任务清单

- 定义训练产物目录规范
  - adapter
  - merged model
  - export logs
  - evaluation results
- 增加导出脚本
  - LoRA adapter 合并到基座
  - 导出为 Hugging Face 权重
  - 转换为 GGUF
- 为导出结果生成 manifest
- 增加 tokenizer 与 prompt template 一致性检查
- 在推理侧增加“加载导出产物”的回归测试
- 支持多版本产物对比
- 记录导出工具版本和命令行参数

### 关键技术点

- adapter merge
- safetensors -> GGUF 转换
- tokenizer 一致性
- prompt template 一致性
- 量化策略选择

### 依赖关系

- 依赖阶段 3 的 GGUF 推理
- 依赖阶段 7 的训练/微调产物

### 完成标志 / 验收标准

- 微调后模型可完成导出并被本地推理加载
- 推理输出能体现微调效果变化
- manifest 能追溯导出来源、基座模型、adapter 版本、量化版本

### 对应当前项目状态评估

**未开始**

### 风险点与注意事项

- GGUF 转换链需要锁定工具版本
- 如果 tokenizer 或 chat template 变化未同步，推理效果会失真
- 不同量化等级会显著影响微调效果保真度

---

## 阶段 9：模型评测、自动化测试与回归体系

### 阶段目标

形成“每次修改都可验证”的工程护栏，覆盖推理、训练、导出和回归。

### 需要完成的具体任务清单

- 扩展单元测试
  - tokenizer
  - runtime config
  - model manifest
- 增加集成测试
  - 模型加载
  - prompt -> generate
  - 会话连续对话
- 增加训练 smoke test
  - 小数据集、少步数
- 增加导出回归测试
  - adapter merge
  - GGUF 可加载性
- 增加评测任务
  - perplexity
  - 指令跟随样例集
  - 基准问答集
- 增加性能回归
  - TTFT
  - tokens/s
  - memory
- 增加 CI 流程
  - C++ build/test
  - Python lint/test
  - smoke inference
- 增加 test fixtures 管理

### 关键技术点

- 小模型/小样本 smoke 方案
- deterministic seed
- golden outputs
- 指标阈值管理
- 测试与基准分层

### 依赖关系

- 横跨阶段 0~8
- 建议从阶段 0 开始逐步补齐，而不是最后一次性补

### 完成标志 / 验收标准

- 每次提交至少跑通单测与最小推理 smoke
- 训练脚本和导出脚本有基本自动化验证
- 有一套可追踪的评测指标基线

### 对应当前项目状态评估

**部分完成**

- 底层单测已存在
- 但没有 tokenizer / runtime / GGUF / e2e / 训练 / 导出测试

### 风险点与注意事项

- 只做底层单测不足以保障产品可用
- 评测样本必须版本化，否则回归结果不稳定

---

## 阶段 10：文档、示例、部署脚本与合规固化

### 阶段目标

把工程从“研发者可理解”提升到“团队可接手、可部署、可合规使用”。

### 需要完成的具体任务清单

- 补全文档
  - 快速开始
  - 模型目录规范
  - GGUF 使用说明
  - 训练说明
  - 导出说明
  - 评测说明
- 增加示例
  - CLI 示例
  - API 示例
  - 本地对话示例
- 增加部署脚本
  - 本地开发环境初始化
  - 推理服务启动
  - 训练环境准备
- 增加许可证与安全边界文档
  - 模型许可证
  - 数据许可证
  - 禁止用途
  - 本地日志与敏感信息保护
- 增加发布说明模板

### 关键技术点

- 文档与代码版本一致
- 环境初始化自动化
- 第三方依赖许可证整理
- 安全边界声明

### 依赖关系

- 横跨全阶段
- 但在阶段 3、7、8 完成后需要系统补齐

### 完成标志 / 验收标准

- 新开发者可按文档完成模型加载、推理、训练、导出
- 每个核心功能都有示例命令
- 许可证和安全约束有明确文档

### 对应当前项目状态评估

**部分完成**

- 有 `README.md`、`docs/ARCHITECTURE.md`
- 但缺少面向使用者和交付的文档体系

### 风险点与注意事项

- 如果没有文档与脚本，训练/推理流程将高度依赖作者记忆
- 许可证问题不能只写在 README 中，必须绑定到模型与数据资产

---

## 当前所处阶段判断

综合仓库现状，当前项目应判断为：

> **处于“阶段 0 未完成 + 阶段 1 未开始”的过渡状态。**

更具体地说：

- 已完成的是：
  - 底层张量、基础算子、KV cache、cache 生命周期管理
- 正在尝试推进但未闭环的是：
  - engine prefill
- 还没有真正开始的是：
  - decode
  - model 抽象
  - tokenizer
  - runtime
  - GGUF 接入
  - CLI / API / 本地服务
  - 训练 / 微调
  - 导出 / 评测 / 合规

### 当前阶段结论

当前项目**不是“已有推理 MVP，准备扩展 GGUF 和训练”**，而是：

> **“底层推理部件 MVP / PoC 已有，但产品级本地推理 MVP 尚未形成。”**

因此路线图的第一优先不是“直接上训练”，而是：

1. 先把当前工程收敛到可持续迭代基线  
2. 再打通最小 GGUF 推理闭环  
3. 最后引入训练与导出链路  

---

## 未完成项清单

## A. 推理基础闭环未完成

- `main.cpp` 为空，缺少程序入口
- `src/runtime/` 为空，缺少 runtime 层
- `src/tokenizer/` 为空，缺少分词能力
- `include/model/layer.h` / `include/model/attention.h` 为空
- `src/model/attention.cpp` 为空
- `include/ops/layernorm.h` 为空，缺少模型必要算子
- `src/engine/decode.cpp` 为空，decode 未实现
- `prefill` 仅完成 KV 写入编排，不是真实模型前向
- 无采样逻辑
- 无 logits 输出
- 无 prompt template 管理

## B. GGUF 支持未开始

- 无 GGUF loader
- 无 GGUF backend adapter
- 无模型 manifest
- 无 tokenizer 兼容处理
- 无模型目录规范
- 无 GGUF 集成测试

## C. 训练/微调未开始

- 无数据接入脚本
- 无清洗流程
- 无统一数据格式
- 无训练配置
- 无训练脚本
- 无 checkpoint / resume
- 无训练日志
- 无导出链路

## D. 工程化能力不足

- 全量构建当前失败
- 缺少 CI
- 缺少集成测试和 E2E
- 缺少部署脚本
- 缺少性能基线
- 缺少文档体系
- 缺少许可证与安全边界说明

---

## 下一步优先级建议

## P0：先修工程基线，再谈功能扩展

### 1. 修复当前构建闭环

优先完成：

- 修复 `CMakeLists.txt` 中 `engine_core -> cache_core` 依赖
- 让 `prefill-test` 真实通过
- 清理 `tests/main_test.cpp` 这类无关内容
- 建立一条本地标准验证命令

这是最优先项，因为当前仓库连“已有能力”都未完全整合成功。

### 2. 把“真实推理 MVP”定义清楚

建议将下一里程碑定义为：

> **加载 1 个 GGUF 模型，输入 prompt，输出生成文本。**

不要把当前“prefill KV 写入”误当作推理 MVP。

### 3. 优先接入成熟 GGUF 后端，而不是自研全链路

推荐优先顺序：

1. 集成 `llama.cpp`
2. 做 backend adapter
3. 做 CLI
4. 再做本地服务

这是从当前代码现实出发的最短路径。

### 4. 训练侧单独开 Python 子模块

建议下一优先设计：

- `python/data_pipeline`
- `python/training`
- `configs/training`

先打通 LoRA / QLoRA 小样本训练，再考虑更多优化。

---

## 风险与依赖

## 1. 技术路线风险

### 1.1 自研推理 runtime 风险高

当前仓库虽然有底层算子，但缺少：

- 权重加载
- tokenizer
- block 级前向
- 采样
- 模型家族兼容
- GGUF 解析

如果坚持完全自研，会显著拉长交付周期。

**建议**：GGUF 推理优先走成熟 backend，当前自研模块继续承担学习、验证和定制扩展职责。

### 1.2 训练与推理格式天然不同

- 训练主格式通常是 HF / safetensors / adapter
- 推理交付格式可以是 GGUF

如果没有 artifact 管理和 manifest，后续会出现：

- 训练产物不可追溯
- 导出结果不可验证
- tokenizer 版本错配
- prompt template 不一致

### 1.3 当前 C++ 工程不适合作为训练主栈

仓库中没有训练图、自动求导、优化器、数据管道基础。

**建议**：训练必须单独采用 Python/PyTorch 栈。

---

## 2. 数据与合规风险

### 2.1 开源数据不等于可任意训练商用

每份数据需要明确：

- 许可证
- 使用范围
- 是否允许商用
- 是否允许再分发
- 是否包含敏感信息

### 2.2 模型许可证风险

GGUF 只是封装格式，不改变原模型许可证。  
必须记录：

- 基座模型 license
- adapter / merge 后产物的 license 继承关系
- 分发限制

### 2.3 本地日志与样本落盘风险

本地训练与推理过程可能写出：

- prompt
- response
- 样本内容
- 调试日志

需要提前定义：

- 默认脱敏策略
- 落盘开关
- 调试日志级别
- 本地缓存清理策略

---

## 3. 工程依赖风险

### 3.1 跨语言栈引入后的维护复杂度

目标架构将包含：

- C++ 推理层
- Python 训练层
- 第三方 backend
- 数据与模型资产管理

需要尽早定义：

- 目录规范
- 版本管理方式
- 一键命令
- 发布与验证流程

### 3.2 平台兼容性风险

本地推理/训练会受以下因素影响：

- Windows / Linux 差异
- 编译器差异
- CUDA 版本
- GPU 驱动
- CPU 指令集

需要尽早明确“首发支持矩阵”。

---

## 4. 推荐里程碑顺序

建议按以下顺序执行，而不是并行铺开：

1. **修复当前工程基线**
2. **明确 runtime / model / tokenizer / backend 抽象**
3. **接入 GGUF 后端并打通最小推理闭环**
4. **提供 CLI / 本地服务**
5. **建立模型 manifest 与配置管理**
6. **建立数据治理链路**
7. **打通 LoRA / QLoRA 训练**
8. **导出 GGUF 并回接推理**
9. **补齐评测、回归、文档、部署、合规**

---

## 最终判断

基于当前仓库实际文件与可验证状态，项目当前最合理的工程定位是：

> **一个已完成底层算子与 KV 缓存验证、正在尝试搭建推理编排层、但尚未进入可用本地大模型推理 MVP 的 C++ 工程。**

因此，要实现“支持 GGUF + 本地训练/微调 + 可持续迭代的训练推理闭环”，最优路线不是继续只在现有算子层堆功能，而是：

- **短期**：先收敛工程基线，快速接入成熟 GGUF 推理后端，做出可用推理 MVP  
- **中期**：用 Python 训练链路补齐 LoRA / QLoRA 和数据治理  
- **长期**：通过模型 manifest、评测、导出、回归和合规体系，把训练与推理串成可持续迭代闭环  

这条路线与当前仓库实际完成度最匹配，落地风险最低，交付速度也最快。