Low-Rank Transformer Language Model Trainer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Efficient Pretraining and Fine-tuning of Low-Rank Transformer Language Models

[Features](#features) • [Quick Start](#quick-start) • [Model Architecture](#model-architecture) • [Training](#training) • [Inference](#inference)

</div>

🚀 Why This Matters

This skeleton addresses three critical challenges in modern LLM development:

1. Computational Efficiency: 70-80% reduction in parameters via low-rank attention
2. Memory Optimization: Progressive sequence length growth & FP8 support
3. Training Stability: ALiBi positional encoding & robust data streaming

Perfect for: Researchers, startups, and developers who need to train capable language models without exascale compute resources.

✨ Key Features

🎯 Core Innovations
- Low-Rank Multi-Head Attention (LoRA-inspired) - Drastically reduces parameter count
- ALiBi Relative Positional Encoding - Better length extrapolation
- Progressive Sequence Length Training - Start small, grow intelligently
- Dual-Phase Training - Pretrain + automatic SFT in one pipeline

⚡ Performance Optimizations
- FP8/BF16/FP16 Mixed Precision - Maximum speed & memory efficiency
- KV Caching - Fast autoregressive generation
- Streaming Datasets - Handle terabytes without loading to RAM
- OOM Recovery - Automatic batch size reduction

🔧 Production Ready
- Hugging Face Integration - Compatible with ecosystem
- Checkpoint Resilience - Resume training from any interruption
- Multi-Dataset Support - Mix & match data sources
- Chat Template Support - Ready for instruction tuning

🏗️ Model Architecture

Low-Rank Attention Mechanism
```python
# Standard MHA: O(d²) parameters
Q = W_q @ x  # [d, d]
K = W_k @ x  # [d, d]  
V = W_v @ x  # [d, d]

# LowRankMHA: O(d·r) parameters (r << d)
Q = (x @ W_q) @ U  # [d, r] where r = 64-96

