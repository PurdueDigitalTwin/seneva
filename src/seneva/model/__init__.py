import torch

# Set the default floating point precision for matmul operations.
torch.set_float32_matmul_precision("high")
