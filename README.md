# AethelStream

> Train frontier-scale LLMs on consumer hardware through sequential layer streaming, predictive I/O overlap, low-rank optimizer compression, and double-pass backward recomputation.

---

## What is AethelStream?

AethelStream is a next-generation neural orchestration framework designed to break the VRAM wall in large-scale model training.

Instead of forcing an entire transformer model to reside in GPU memory, AethelStream treats the GPU as a high-speed compute cache. Only the active transformer layer exists in VRAM at any moment while all other layers are streamed dynamically from NVMe SSD through RAM.

The system combines:

* Sequential Layer Streaming (SLS)
* Double-pass backward recomputation
* Predictive asynchronous I/O
* GaLore low-rank optimizer projection
* LOMO fused updates
* MemAscend-inspired memory virtualization
* Streaming-aware mixed precision execution

The goal is simple:

> Make training and fine-tuning 70B+ parameter models possible on consumer hardware.

---

## Why This Exists

Modern LLM training assumes:

* Entire model must live in VRAM
* Optimizer states remain resident
* Activations are stored globally
* GPUs are the primary memory source

This completely breaks on consumer systems.

Example:

| Model | Approx Training Memory |
| ----- | ---------------------- |
| 7B    | ~84 GB                 |
| 13B   | ~156 GB                |
| 70B   | ~840 GB                |

A single RTX 4090 has only 24 GB VRAM.

AethelStream changes the paradigm from:

### Spatial Residency

"How much VRAM do you have?"

to:

### Temporal Streaming

"How efficiently can you move layers through the compute pipeline?"

---

## Core Idea

Traditional frameworks keep all layers resident.

AethelStream streams them dynamically.

### Forward Pass

```text
SSD → RAM → VRAM → Compute → Evict
```

### Backward Pass

Instead of storing every activation:

* sparse checkpoints are saved
* activations are recomputed locally during backward
* gradients are projected into low-rank space
* optimizer updates happen immediately
* gradients are discarded instantly

This keeps memory usage nearly constant regardless of model scale.

---

# Architecture Overview

## 1. Sequential Layer Streaming Engine

Only one transformer block exists in VRAM at a time.

Pipeline:

```text
SSD → CPU Transit Buffer → GPU → Compute → Evict
```

Features:

* deterministic streaming
* pinned-memory transit buffers
* asynchronous prefetching
* sliding window buffering
* delayed write-back
* streaming-aware execution scheduling

---

## 2. Double-Pass Backward Algorithm

Backward propagation normally requires storing all activations.

AethelStream avoids this by recomputing activations locally.

Process:

1. Save sparse checkpoints during forward pass
2. During backward:

   * locate nearest checkpoint
   * run mini-forward recomputation
   * regenerate local activations
   * compute gradients
   * update weights instantly
   * discard gradients immediately

This reduces activation memory dramatically while preserving gradient parity.

---

## 3. Predictive I/O Overlap

The access pattern of transformers is deterministic.

That means the system always knows which layer will be needed next.

AethelStream overlaps:

* NVMe reads
* PCIe transfers
* GPU compute
* SSD write-back

simultaneously.

At runtime:

```text
SSD reading layer i+2
RAM transferring layer i+1
GPU computing layer i
```

The goal is to keep GPU idle time near zero.

---

## 4. GaLore Optimizer Compression

Optimizer states are one of the largest memory bottlenecks.

AethelStream applies low-rank gradient projection:

R_t = P_t^{T} G_t Q_t

Gradients are projected into compact low-rank space, updated there, then reconstructed.

This allows:

* compressed optimizer states
* 8-bit optimizer storage
* dramatically lower RAM usage
* scalable full-parameter training

---

## 5. MemAscend-Inspired Memory Virtualization

Custom memory management replaces PyTorch's default allocators.

Includes:

* alignment-free pinned memory
* adaptive buffer pools
* direct NVMe I/O
* io_uring asynchronous reads
* fragmentation-aware allocation
* fused overflow checking

The framework is designed around systems-level control rather than Python runtime abstractions.

---

## 6. Precision-Aware Streaming

Not all layers require identical precision.

AethelStream streams:

* sensitive layers in FP16
* middle layers in INT4
* adapters in FP16

This reduces PCIe bandwidth while maintaining convergence quality.

---

# Key Features

* Train models larger than VRAM capacity
* Consumer GPU support
* Streaming-first architecture
* Linux optimized runtime
* Rust systems core
* CUDA fused kernels
* Hugging Face compatibility
* LoRA support
* MoE-aware routing support
* Dynamic checkpoint optimization
* Numerical parity guard
* Adaptive hardware profiling

---

# Technical Stack

| Component        | Technology   |
| ---------------- | ------------ |
| Core Runtime     | Rust         |
| GPU Kernels      | CUDA C++     |
| Python Interface | PyO3         |
| Async Runtime    | tokio        |
| NVMe I/O         | io_uring     |
| Tensor Format    | safetensors  |
| Fused Kernels    | Triton       |
| Quantization     | bitsandbytes |

---

# Target Hardware

Designed for consumer systems:

| Component | Target              |
| --------- | ------------------- |
| GPU       | RTX 3060 → RTX 4090 |
| VRAM      | 8 GB → 24 GB        |
| RAM       | 64 GB → 128 GB      |
| Storage   | NVMe SSD            |
| OS        | Linux               |

Linux is the primary high-performance path.

---

# Performance Goals

| Metric                 | Target            |
| ---------------------- | ----------------- |
| 70B Training VRAM      | < 22 GB           |
| 7B Training on 8GB GPU | < 7 GB            |
| GPU Idle Time          | < 20%             |
| Context Length         | 8K+               |
| Gradient Parity        | within 1e-5       |
| Optimizer RAM          | < 5 GB compressed |

---

# Current Development Status

AethelStream is currently in active research and implementation.

Development is organized into modular subsystems:

* Model Sharding Engine
* Memory Manager
* Prefetch Pipeline
* Optimizer Compression
* Double-Pass Backward Engine
* LoRA Adapter Manager
* Streaming Kernels
* Numerical Validation Systems

The framework follows a spiral development model focused on validating each subsystem independently before full integration.

---

# Long-Term Vision

AethelStream is not just an optimization framework.

It is an attempt to rethink how large-scale neural systems are trained.

The project explores a future where:

* frontier model training becomes accessible
* compute becomes temporal rather than spatial
* memory virtualization replaces brute-force hardware scaling
* consumer hardware participates in frontier AI research

---

# Research Direction

Primary inspirations and integrated concepts include:

* GaLore
* LOMO
* MeSP
* LoHan / Fuyou
* MemAscend
* AirLLM
* ZeRO-Infinity

AethelStream extends these ideas into a unified streaming-based training architecture.

---

# Repository Structure

```text
aethelstream/
├── ramflow/
├── shard_engine/
```

---

# Status

🚧 Research & Development
⚡ Linux-first
🧠 Frontier-scale training research
🦀 Rust systems architecture
🔥 Consumer hardware focused

---

# Research Document

The complete architecture, algorithms, module breakdown, and implementation plan are documented in the internal ideation and PoA document. 
