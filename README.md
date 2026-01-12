<div align="center">

# EdgeNet MLIR Deployment
### End-to-End AI Compiler Pipeline for Embedded Systems

![Edge AI](https://img.shields.io/badge/Edge-AI-107C10?style=flat-square)
![Compiler](https://img.shields.io/badge/Domain-AI%20Compiler-005A9C?style=flat-square)
![Target](https://img.shields.io/badge/Target-Embedded%20Systems-333333?style=flat-square)

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-red?style=flat-square)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-green?style=flat-square)](https://onnx.ai/)
[![MLIR](https://img.shields.io/badge/MLIR-orange?style=flat-square)](https://mlir.llvm.org/)
[![LLVM](https://img.shields.io/badge/LLVM-lightgrey?style=flat-square)](https://llvm.org/)
[![C](https://img.shields.io/badge/C-blueviolet?style=flat-square)](https://en.wikipedia.org/wiki/C_(programming_language))
[![Linux](https://img.shields.io/badge/Linux-Ubuntu-black?style=flat-square)](https://ubuntu.com/)

</div>

---

## Executive Summary

**EdgeNet MLIR Deployment** is an end-to-end proof of concept demonstrating how a neural network can be transformed from a high-level training framework into **optimized, deployable artifacts** for embedded and edge-class systems using modern AI compiler infrastructure.

The project intentionally prioritizes **compiler workflows**, **intermediate representations**, and **deployment constraints** over model accuracy or dataset performance.

The structure and tooling mirror real-world pipelines used in **semiconductor companies**, **edge AI toolchains**, and **AI compiler teams**.

---

## Project Focus

This repository is explicitly compiler-driven:

- End-to-end ownership of an AI compilation pipeline
- Explicit use of ONNX, MLIR, and LLVM intermediate representations
- Embedded and edge deployment constraints as first-class concerns
- Clear separation between ML framework, compiler, and runtime layers

Accuracy is not the goal. **Deployability is.**

---

## Motivation

Deploying machine learning models on embedded systems is fundamentally a compiler problem.

Key constraints addressed:
- Deterministic execution
- Limited memory and bandwidth
- Hardware heterogeneity (CPU, DSP, NPU)
- Toolchain and backend integration boundaries

This project was built to understand these constraints from a **compiler and systems perspective**, not as a black-box ML workflow.

---

## High-Level Architecture

PyTorch Model

↓

ONNX Export

↓

ONNX-MLIR

↓

MLIR Dialects (TOSA)

↓

MLIR Optimization Passes

↓

LLVM IR

↓

Executable / Embedded C Integration


Each stage represents a real production boundary in modern edge AI software stacks.

---

## Technologies Used

- **PyTorch** – Model definition and training
- **ONNX** – Framework-agnostic model interchange
- **MLIR** – Multi-level intermediate representation for AI compilers
- **LLVM** – Backend code generation
- **C** – Embedded integration layer
- **Linux (Ubuntu via WSL)** – Development and build environment

---

## Project Structure

```bash
edgenet-mlir/
├── src/
│   ├── train.py                 # Compact CNN training
│   ├── export_to_onnx.py        # PyTorch → ONNX export
│   ├── convert_to_mlir.sh       # ONNX → MLIR lowering
│   ├── optimize_mlir.sh         # MLIR optimization passes
│   ├── generate_c_code.sh       # MLIR → LLVM → executable
│   └── run_inference_test.c     # Embedded integration stub
│
├── models/
│   └── mnist_cnn.onnx           # Exported neural network
│
└── mlir/
    ├── model.mlir               # Initial MLIR representation
    └── model_opt.mlir           # Optimized MLIR representation
```

## Design Principles

Models are intentionally small and static

Graphs are compiler-friendly

Shapes are fully known at compile time

Abstractions mirror real deployment toolchains

The goal is to study representation, lowering, and code generation, not to chase benchmark accuracy.

## Prerequisites

Python 3.9+

PyTorch

ONNX

ONNX-MLIR toolchain

LLVM / Clang

Linux environment (tested on Ubuntu via WSL)

## Execution
```bash
cd src
python3 train.py
python3 export_to_onnx.py
bash convert_to_mlir.sh
bash optimize_mlir.sh
bash generate_c_code.sh
```

## What This Project Demonstrates

End-to-end AI compiler pipeline ownership

MLIR dialect-based lowering and optimization

Embedded-oriented model and compilation design

Backend code generation awareness

Clear integration boundaries between ML frameworks, compilers, and embedded software

## Project Alignment

This work aligns with roles and domains in:

AI Compiler Engineering

Edge AI Software

Embedded Systems and Semiconductor Toolchains

## Known Limitations

No hardware-specific backend

Embedded execution is stubbed

No quantization or calibration passes

These are intentional trade-offs to maintain focus on compiler fundamentals.

## Roadmap

INT8 quantization and calibration

Custom MLIR dialect for a simulated NPU

ARM Cortex-M backend generation

Performance and memory profiling

QEMU-based embedded execution

