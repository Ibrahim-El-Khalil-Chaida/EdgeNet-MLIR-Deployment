import torch
from train import SmallCNN

# Load trained model
model = SmallCNN()
model.load_state_dict(torch.load("../models/cnn.pt"))
model.eval()

# Dummy input defines static shape for compiler pipeline
dummy_input = torch.randn(1, 1, 28, 28)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "../models/mnist_cnn.onnx",
    opset_version=11
)

print("Model exported to ONNX.")
