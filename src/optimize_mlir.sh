#!/bin/bash
set -e

# Apply MLIR optimization and dialect lowering
mlir-opt ../mlir/model.mlir \
  --convert-onnx-to-tosa \
  --tosa-optimize \
  > ../mlir/model_opt.mlir

echo "MLIR optimization completed."
