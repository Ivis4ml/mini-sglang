# Mini-SGLang 学习指南

## 概述

这是一个 **Bottom-Up** 的 Mini-SGLang 学习指南，帮助你从基础概念开始，逐步深入理解这个高性能 LLM 推理框架的核心实现。

## 项目结构

```
mini-sglang/
├── README.md                          # 项目主文档
├── docs/
│   ├── features.md                    # 功能说明
│   └── structures.md                  # 架构文档
├── python/minisgl/                    # 源代码 (~6000 行)
│   ├── core.py                        # 核心数据结构
│   ├── engine/                        # 推理引擎
│   ├── scheduler/                     # 调度器
│   ├── attention/                     # Attention 后端
│   ├── kvcache/                       # KV Cache 管理
│   ├── layers/                        # 模型层
│   ├── models/                        # 模型实现
│   └── server/                        # API 服务器
└── learning_guide/                    # 本学习指南
    └── notebooks/                     # Jupyter Notebooks
```

## 学习路径

### Module 0: PyTorch 前置知识 (新增!)
**文件**: `notebooks/00_pytorch_prerequisites.ipynb`

- Tensor 基础操作 (创建、索引、形状变换)
- CUDA 设备管理与同步
- **CUDA Stream** (Overlap Scheduling 的基础)
- **CUDA Graph** (Decode 加速的关键)
- nn.Module 模型结构
- torch.distributed 分布式通信
- 内存管理 (Pinned Memory, Contiguous)
- 数据类型 (bfloat16)

### Module 1: LLM 推理基础
**文件**: `notebooks/01_llm_inference_basics.ipynb`

- LLM 推理的两个阶段 (Prefill/Decode)
- KV Cache 的作用和重要性
- 为什么需要推理框架

### Module 2: 核心数据结构
**文件**: `notebooks/02_core_data_structures.ipynb`

- `SamplingParams`: 采样参数
- `Req`: 请求表示
- `Batch`: 批处理
- `Context`: 全局上下文
- Page Table 详解

### Module 3: 模型层
**文件**: `notebooks/03_model_layers.ipynb`

- Tensor Parallelism 基础
- Linear 层的并行实现
- RMSNorm 及其融合优化
- Rotary Position Embedding (RoPE)

### Module 4: Attention 机制
**文件**: `notebooks/04_attention_mechanism.ipynb`

- 标准 Attention 的计算过程
- FlashAttention 优化原理
- Paged Attention 概念
- Mini-SGLang 的 Attention Backend 设计

### Module 5: KV Cache 管理
**文件**: `notebooks/05_kv_cache_management.ipynb`

- Naive Cache Manager
- Radix Cache 核心概念
- 前缀匹配和缓存复用
- LRU 驱逐策略

### Module 6: Scheduler 调度逻辑
**文件**: `notebooks/06_scheduler_logic.ipynb`

- Scheduler 的核心职责
- Prefill 和 Decode 批处理
- Chunked Prefill 策略
- Continuous Batching

### Module 7: 完整推理流程
**文件**: `notebooks/07_inference_flow.ipynb`

- 请求生命周期完整跟踪
- Engine 和 Scheduler 的协作
- 多进程通信机制
- Overlap Scheduling 详解

### Module 8: 高级优化
**文件**: `notebooks/08_advanced_optimizations.ipynb`

- CUDA Graph 原理和应用
- Tensor Parallelism 实现
- FlashAttention/FlashInfer 集成
- 性能调优建议

### Module 9: 深度设计分析
**文件**: `notebooks/09_deep_analysis.ipynb`

- page_size=1 的设计决策分析
- 残差连接融合 (RMSNormFused)
- CUDA Graph Pool 内存复用机制
- Batch Padding 策略与效率
- 采样温度处理细节
- 多进程通信架构 (ZMQ)
- 二级内存管理设计
- GQA (Grouped Query Attention) 支持
- Stream 同步与 Overlap 调度
- Weight Tying 节省内存
- NVTX Profiling 性能分析

## 在 Google Colab 中使用

### 方法 1: 直接打开

1. 访问 Google Colab: https://colab.research.google.com
2. 选择 "GitHub" 标签页
3. 输入本仓库的 URL
4. 选择对应的 notebook 文件

### 方法 2: 克隆仓库

```python
# 在 Colab 中运行
!git clone https://github.com/sgl-project/mini-sglang.git
%cd mini-sglang/learning_guide/notebooks
```

### 方法 3: 本地运行

```bash
# 克隆仓库
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install jupyter torch matplotlib

# 启动 Jupyter
jupyter notebook learning_guide/notebooks/
```

## 重点学习内容

### 核心概念

| 概念 | 说明 | 重要性 |
|------|------|--------|
| KV Cache | 缓存 K/V 避免重复计算 | ⭐⭐⭐⭐⭐ |
| Prefill/Decode | 推理的两个阶段 | ⭐⭐⭐⭐⭐ |
| Paged Attention | 分页管理 KV Cache | ⭐⭐⭐⭐⭐ |
| Radix Cache | 前缀复用优化 | ⭐⭐⭐⭐ |
| Continuous Batching | 动态批处理 | ⭐⭐⭐⭐ |
| CUDA Graph | 减少 CPU 开销 | ⭐⭐⭐ |
| Tensor Parallelism | 多 GPU 并行 | ⭐⭐⭐ |

### 关键文件

| 文件 | 内容 |
|------|------|
| `core.py` | `Req`, `Batch`, `Context` 定义 |
| `engine/engine.py` | 推理引擎主逻辑 |
| `scheduler/scheduler.py` | 调度器主循环 |
| `kvcache/radix_manager.py` | Radix Cache 实现 |
| `attention/fi.py` | FlashInfer 后端 |

### 推荐阅读顺序

0. **前置**: Module 0 (PyTorch 基础，如已熟悉可跳过)
1. **入门**: Module 1-2 (理解基本概念)
2. **进阶**: Module 3-5 (深入核心组件)
3. **高级**: Module 6-8 (掌握优化技术)
4. **深入**: Module 9 (设计决策和实现细节)

## 动手实践建议

1. **阅读代码**: 每个模块都有对应的源代码链接
2. **修改参数**: 尝试不同的配置看效果
3. **添加日志**: 在关键位置添加 print 语句
4. **运行 Benchmark**: 使用 `benchmark/` 目录的脚本测试性能

## 常见问题

### Q: 需要 GPU 吗?
A: 大部分概念可以在 CPU 上理解，但实际运行 Mini-SGLang 需要 NVIDIA GPU。

### Q: 支持哪些模型?
A: 目前支持 Llama-3 和 Qwen-3 系列。

### Q: 如何贡献?
A: 欢迎提交 Issue 或 PR 到 [SGLang 仓库](https://github.com/sgl-project/sglang)。

## 参考资源

- [Mini-SGLang 主文档](../README.md)
- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [FlashInfer 项目](https://github.com/flashinfer-ai/flashinfer)
- [vLLM 论文 (PagedAttention)](https://arxiv.org/abs/2309.06180)

---

Happy Learning! 🚀
