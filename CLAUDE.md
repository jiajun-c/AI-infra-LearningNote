# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

AI-infra-LearningNote is a comprehensive learning repository for AI infrastructure knowledge, covering:

- **CUDA Programming**: GPU architecture, primitives, TensorCore, memory optimization, Hopper architecture features
- **Programming Languages**: C++, Python, Triton
- **Large Language Models**: Architecture (Attention, MoE, Linear layers), parallel training (DP, DDP, TP, PP)
- **Training & Inference Optimization**: Quantization, pruning, fine-tuning, kv-cache, batch strategies
- **Model Communication**: Communication backends (gloo, NCCL), collective primitives
- **Agent**: LangChain, inference, vector databases
- **Performance Analysis**: Profiling, debugging tools

## File Structure

```
├── cuda/           # CUDA programming notes and code
├── lang/           # Programming language notes (cpp, python, Triton)
├── LLMArch/        # LLM architecture components
├── parallel/       # Parallel training strategies
├── quant/          # Quantization methods
├── inference/      # Inference optimization techniques
├── comm/           # Communication libraries and primitives
├── framework/      # Deep learning frameworks (PyTorch, DeepSpeed)
├── agent/          # AI agent related notes
├── tools/          # Development tools and utilities
└── xpu/            # XPU architecture notes
```

## Code Conventions

- Each topic has its own directory with a `README.md` for documentation
- Code examples are placed alongside documentation in topic directories
- Follow existing naming conventions in the repository

## Common Tasks

- When adding new topics, create a new directory with a `README.md`
- Use relative links for cross-references in markdown files
- Keep documentation concise and focused on learning notes

## Language

This repository primarily uses Chinese for documentation. When editing or creating documentation files, prefer Chinese for consistency.