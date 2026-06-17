# 术语表

本文解释 qwen_cpp 文档和源码中常见术语。术语按项目语境说明，不追求覆盖全部深度学习背景。

## Tensor

项目中的基础张量容器。当前实现使用连续 row-major 存储，支持 N 维 shape/stride 信息，同时保留二维矩阵访问接口。大多数算子、缓存和模型组件都以 `Tensor` 作为数据载体。

## Row-major

行优先内存布局。二维矩阵中，同一行的元素在内存中连续存放。项目中的矩阵访问和 append 行为都基于这一假设。

## ops

无状态计算算子层。当前包括 `matmul`、`softmax` 和 attention。ops 只处理输入、配置和输出，不保存跨调用状态。

## Attention

Transformer 中用于根据 query、key、value 计算上下文表示的机制。当前项目实现 scaled dot-product attention，并支持 additive mask、causal mask、GQA 和 RoPE。

## Causal mask

自回归生成中使用的上三角屏蔽规则。它阻止当前位置看到未来 token。decode 场景中需要结合 `query_position_offset` 表达 query 在完整序列中的绝对位置。

## Additive mask

加到 attention score 上的 mask。常见写法是可见位置加 0，不可见位置加一个很大的负数。项目中 additive mask 与 causal mask 可以叠加。

## GQA

Grouped Query Attention。query head 数可以大于 key/value head 数，多个 query head 共享同一个 KV head。Qwen 系列模型常用这种结构来降低 KV cache 成本。

## RoPE

Rotary Position Embedding，旋转位置编码。它把位置信息注入 query/key 表示。项目 attention 配置中包含 `rope_theta` 和 `rope_scale`。

## KV Cache

Key/Value 缓存。自回归推理中，历史 token 的 key/value 可以缓存起来，decode 新 token 时只追加当前步 KV，避免重复计算完整历史序列。

## Prefill

推理的 prompt 处理阶段。模型一次性处理输入 prompt，并把多层历史 KV 写入缓存。当前项目的 `PrefillEngine` 只负责把外部提供的 KV 写入 `KVCache`，不负责完整模型前向。

## Decode

自回归生成阶段。模型每次处理一个新 token，并把当前步 KV 追加到缓存。当前项目的 `DecodeEngine` 负责单步 KV 追加编排。

## CacheManager

管理多个 `KVCache` 实例的组件。它提供以 cache id 为入口的缓存生命周期管理，供 engine 层使用。

## ModelConfig

模型结构配置。当前记录模型家族、词表大小、层数、hidden size、attention head 数、KV head 数、RoPE 参数等字段，并提供基础合法性检查。

## ModelWeights

模型权重容器。当前支持 Qwen 风格最小模型组件所需权重的组织和 shape 检查。真实 GGUF/manifest 权重加载尚未实现。

## QwenModel

项目中的 Qwen 风格最小模型骨架。它可以基于 token embedding 执行若干 TransformerBlock 的前向，并计算 logits，但当前还没有 tokenizer、sampling 和真实模型加载闭环。

## Runtime

未来的推理运行时封装层。它应负责模型加载、session 管理、generation config、sampling config 和 backend 调用。当前仍是脚手架。

## Backend

未来对接真实模型推理后端的适配层。当前 `GgufLlamaCppBackend` 表明项目倾向接入 GGUF/llama.cpp，但真实加载和上下文创建尚未实现。

## GGUF

一种本地大模型推理常用的模型文件格式，常与 llama.cpp 生态配合使用。项目计划支持 GGUF，但当前没有真实加载能力。

## LoRA / QLoRA

参数高效微调方法。项目中已有训练配置和 adapter 目录约定，但训练脚本尚未实现。

## Manifest

模型 artifact 清单。它应记录模型 id、模型家族、文件路径、tokenizer、license、hash、来源和导出信息。当前只有示例文件，runtime 还未消费 manifest。

## Scaffold

脚手架。指目录、头文件或函数入口已经存在，但没有真实可用实现，或尚未接入构建、测试和文档闭环。qwen_cpp 中 tokenizer、runtime、backend、CLI、service 和 Python 训练导出目前都应按脚手架理解。
