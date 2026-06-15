# qwen_cpp

一个面向 Transformer 推理路径的 C++ 学习型工程，当前主线聚焦：
- `Tensor` 基础 N 维张量容器
- `ops` 纯计算算子（`matmul / softmax / attention`）
- `cache` KV 缓存与分配管理
- `engine` 中的 KV 编排（`prefill / decode`）
- `model` 中的 Qwen 风格 RMSNorm / GQA Attention / SwiGLU MLP / TransformerBlock 骨架
- 可按 label 分类运行的 CTest 测试

> 目标：以 `tensor -> ops -> cache -> engine -> model` 为学习主线，逐步理解 Transformer 推理路径中的关键抽象与边界。

---

## 目录结构

- `include/`：头文件
  - `ops/`：计算算子接口
  - `cache/`：缓存与生命周期管理接口
  - `engine/`：prefill / decode 编排接口
  - `model/`：模型层级抽象与实验模块
- `src/`：实现文件
  - `ops/`：无状态算子实现
  - `cache/`：KVCache / CacheManager / CacheAllocator
  - `engine/`：已实现的 KV 编排逻辑
- `tests/`：CTest 可执行测试与分类说明

## 主线与支线

- 学习主线：`tensor -> ops -> cache -> engine -> model`
- 当前主入口：测试，而不是 CLI 或服务
- 实验支线：`tokenizer / runtime / backend / cli / service`
- 实验支线代码当前保留为脚手架，用于后续学习，不作为当前主架构完成度标准

---

## 构建与运行

### 1) 配置与构建

```powershell
cmake -S . -B build
cmake --build build
```

### 2) 运行测试

```powershell
ctest --test-dir build --output-on-failure
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L integration --output-on-failure
```

### 3) 运行核心路径基准

```powershell
.\build\benchmarks\qwen_decode_bench.exe
.\build\benchmarks\qwen_decode_bench.exe --prompt 128 --decode 64 --layers 4 --q-heads 8 --kv-heads 2 --head-dim 16
```

该基准覆盖 GQA attention + RoPE + 增量 causal mask + KVCache 追加/容量利用率，用于观察 decode 路径的吞吐趋势。

---

## 架构职责（核心约束）

> 详细说明见：`docs/ARCHITECTURE.md`

- **ops：纯计算，无状态优先**
  - 只关注输入 -> 输出，不持有跨调用状态
  - 不管理缓存生命周期
- **cache：状态与生命周期**
  - 负责 KV 的创建、增长、回收、容量约束
  - 对 engine 暴露可控状态接口
- **engine：流程编排**
  - 负责 prefill/decode 执行顺序
  - 串联 model、ops、cache，不做底层算子细节
- **model：层级抽象**
  - 作为学习阶段的下一层抽象，逐步定义 Block/Layer 等模型结构
  - 当前包含少量实验模块，不要求形成完整产品接口

---

## 当前状态

- ✅ `ops` 和 `cache` 已有可运行实现与测试
- ✅ `engine` 已完成 `prefill / decode` 的 KV 编排与测试
- ✅ `model` 已具备最小可编译 Qwen block/model 前向骨架与测试
- 🚧 `tokenizer / runtime / backend / cli / service` 为实验支线脚手架

建议下一步：
1. 补一条从 `model` 到 `engine/cache` 的教学型集成用例。
2. 接入真实 GGUF/llama.cpp backend 或明确自研权重加载格式。
3. 继续把实验支线与学习主线的边界写清楚，避免文档漂移。
