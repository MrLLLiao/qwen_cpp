# qwen_cpp

一个面向 Transformer 推理路径的 C++ 学习型工程，当前包含：
- `Tensor2D` 基础张量容器
- `ops` 纯计算算子（`matmul / softmax / attention`）
- `cache` KV 缓存与分配管理
- 基础单测（可执行测试）

> 目标：逐步从“算子可用”演进到“可编排推理引擎（prefill/decode）”。

---

## 目录结构

- `include/`：头文件
  - `ops/`：计算算子接口
  - `cache/`：缓存与生命周期管理接口
  - `model/`：模型层级抽象接口（当前为占位）
- `src/`：实现文件
  - `ops/`：无状态算子实现
  - `cache/`：KVCache / CacheManager / CacheAllocator
  - `engine/`：流程编排入口（当前为占位）
  - `model/`：模型层实现（当前为占位）
- `tests/`：可执行单测

---

## 构建与运行

### 1) 配置与构建

```powershell
cmake -S . -B build
cmake --build build
```

### 2) 运行测试

```powershell
./build/tensor-test.exe
./build/matmul-test.exe
./build/softmax-test.exe
./build/attention-test.exe
./build/kvcache-test.exe
```

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
  - 定义 Block/Layer 等模型结构与参数访问
  - 对 engine 提供稳定模型语义接口

---

## 当前状态

- ✅ `ops` 和 `cache` 已有可运行实现与测试
- 🚧 `engine` 与 `model` 仍在搭建（当前多为占位文件）

建议下一步：
1. 在 `model` 中落地最小 `AttentionLayer` 抽象。
2. 在 `engine/prefill.cpp` 中打通一次完整前向（含 KV append）。
3. 在 `engine/decode.cpp` 中实现单步增量解码。
