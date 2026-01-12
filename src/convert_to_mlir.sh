#!/bin/bash
set -e

# Convert ONNX model to MLIR
onnx-mlir ../models/mnist_cnn.onnx

mv mnist_cnn.mlir ../mlir/model.mlir
echo "ONNX successfully lowered to MLIR."
