# 架构职责文档

本文档用于明确项目分层边界，避免“算子、状态、流程、语义”耦合。

## 1. 分层总览

```
engine (流程编排)
   ↓ 调用
model  (层级抽象)
   ↓ 依赖
ops    (纯计算)
   ↘
    cache (状态与生命周期，受 engine 驱动)
```

关键原则：
- **ops 不持久化状态**
- **cache 不做流程决策**
- **engine 不实现底层数学细节**
- **model 不直接管理全局生命周期**

---

## 2. 职责定义

### 2.1 ops：纯计算，无状态优先

**职责**
- 提供确定性计算：`matmul`、`softmax`、`attention` 等。
- 接受输入张量与配置，返回输出张量。
- 进行必要的输入合法性校验（shape、参数范围）。

**非职责**
- 不保存跨 step 的缓存。
- 不感知 session/request 生命周期。
- 不决定 prefill/decode 顺序。

**代码位置**
- `include/ops/*`
- `src/ops/*`

---

### 2.2 cache：状态与生命周期

**职责**
- 管理 KV 状态：初始化、追加、查询、释放。
- 管理内存资源池与容量约束（max_tokens / max_buffers）。
- 提供按层、按区间的只读视图（例如 `Tensor2DView`）。

**非职责**
- 不做 attention 数学计算。
- 不决定何时 append（由 engine 决策）。
- 不定义模型层语义。

**代码位置**
- `include/cache/*`
- `src/cache/*`

---

### 2.3 engine：流程编排

**职责**
- 编排推理阶段：
  - prefill：批量写入历史 token 的 KV。
  - decode：逐 token 增量推进。
- 驱动 model 前向执行，协调 cache 读写。
- 管理一次请求/会话级别的执行上下文。

**非职责**
- 不实现底层算子细节（交给 ops）。
- 不拥有模型结构定义（交给 model）。
- 不替代 cache 做内存池细节。

**代码位置**
- `src/engine/prefill.cpp`
- `src/engine/decode.cpp`

> 当前状态：占位，待逐步落地。

---

### 2.4 model：层级抽象

**职责**
- 描述模型组件及层级关系（Layer / Block / AttentionLayer）。
- 定义参数访问与前向接口语义。
- 屏蔽底层实现差异，为 engine 提供稳定 API。

**非职责**
- 不直接实现通用算子（交给 ops）。
- 不接管全局缓存生命周期（交给 cache + engine）。

**代码位置**
- `include/model/*`
- `src/model/*`

> 当前状态：占位，待补全接口与实现。

---

## 3. 依赖方向（必须遵守）

允许依赖：
- `engine -> model`
- `engine -> cache`
- `engine -> ops`（必要时）
- `model -> ops`

不允许依赖：
- `ops -> cache/engine/model`
- `cache -> engine/model/ops业务逻辑`
- `model -> engine`

一句话：**底层可被上层调用，上层不要反向被底层感知。**

---

## 4. 协作契约（接口级）

- ops 接口应保持“函数式 + 配置对象”风格，避免隐式状态。
- cache 接口应显式表达容量、索引、越界行为（抛异常或错误码）。
- engine 在每个阶段只做一件事：组织调用与状态推进。
- model 接口优先稳定语义（如 `forward` 输入输出约定），再考虑内部实现。

---

## 5. 演进建议

1. 定义 `model` 最小可用接口（如 `Layer::forward`，`AttentionLayer`）。
2. 在 `engine` 增加请求上下文结构（session id、step、cache id）。
3. 以端到端用例串联：`prefill -> decode(1 step)`，补齐集成测试。
4. 若后续引入并行/异步，仍保持本职责边界不变。
