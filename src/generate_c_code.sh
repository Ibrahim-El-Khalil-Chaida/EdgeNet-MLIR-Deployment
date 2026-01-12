#!/bin/bash
set -e

# Translate MLIR to LLVM IR
mlir-translate ../mlir/model_opt.mlir --mlir-to-llvmir > model.ll

# Compile LLVM IR
clang model.ll -O2 -o inference_bin

echo "Executable generated."
