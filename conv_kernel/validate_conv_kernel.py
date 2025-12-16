from conv_kernel_scaffold import naive_conv_kernel

import torch
import torch.nn.functional as F

# Input tensor

batch_size = 1
in_channels = 3
in_height = 8
in_width = 8
in_tensor = torch.randn(batch_size, in_channels, in_height, in_width)

out_channels = 64
filter_height = 3
filter_width = 3
filter_tensor = torch.randn(out_channels, in_channels, filter_height, filter_width)

stride = 1
out_height = int(1 + (in_height - filter_height) / stride)
out_width = int(1 + (in_width - filter_width) / stride)
out_tensor_mine = torch.randn(batch_size, out_channels, out_height, out_width)

out_tensor_gt = F.conv2d(in_tensor, filter_tensor, stride=[stride, stride])
out_tensor_mine = naive_conv_kernel(in_tensor, filter_tensor, stride, out_tensor_mine)

print("GT convolution:")
print(out_tensor_gt)
print("\nMy convolution:")
print(out_tensor_mine)

print("\nValidation:", end=" ")
if torch.allclose(out_tensor_gt, out_tensor_mine, atol=1e-5, rtol=1e-7) and out_tensor_gt.shape == out_tensor_mine.shape:
    print("OK")
else:
    print("Fail")

print("\nDebug:")
print(f"GT Shape: {out_tensor_gt.shape}")
print(f"Mine Shape: {out_tensor_gt.shape}")
print(f"Max Difference: {(out_tensor_gt - out_tensor_mine).abs().max()}")

