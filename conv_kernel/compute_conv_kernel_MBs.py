from conv_kernel_config import *
    
# ResNet-50 Architecture
# Input: 224x224x3
# conv1: 7x7 conv, 3->64, stride 2

conv1 = ConvBlock(
    TensorShape(224,224,3),
    [Filter(7,7,64,2)]
)

# After conv1: 112x112x64
# After maxpool (3x3, stride 2): 56x56x64

conv2 = ConvBlock(
    TensorShape(56,56,64),
    [
        Filter(1,1,64), 
        Filter(3,3,64), 
        Filter(1,1,256)
    ] * 3
)

conv3 = ConvBlock(
    conv2.out_tensor,
    [
        Filter(1,1,128), 
        Filter(3,3,128), 
        Filter(1,1,512)
    ] * 4,
    downsample=True
)

conv4 = ConvBlock(
    conv3.out_tensor,
    [
        Filter(1,1,256), 
        Filter(3,3,256), 
        Filter(1,1,1024)
    ] * 6,
    downsample=True
)

conv5 = ConvBlock(
    conv4.out_tensor,
    [
        Filter(1,1,512), 
        Filter(3,3,512), 
        Filter(1,1,2048)
    ] * 3,
    downsample=True
)

example_conv = Conv(
    in_tensor=TensorShape(226, 226, 3),
    filter=Filter(h=3, w=3, out_ch=64),
    pad=False
)

print(example_conv.get_naive_ms())
print(example_conv.get_cudnn_ms())

print()

all_conv_blocks = [
    conv1,
    conv2,
    conv3,
    conv4,
    conv5
]

print(conv1)
print(conv2)
print(conv3)
print(conv4)
print(conv5)

print("=" * 50)
print("Full CNN:")
print("=" * 50)
print(f"Total naive MBs: {sum(conv.get_naive_MBs() for conv in all_conv_blocks):.2f}")
print(f"Total naive ms: {sum(conv.get_naive_ms() for conv in all_conv_blocks):.2f}")
print(f"\nTotal cudnn MBs: {sum(conv.get_cudnn_MBs() for conv in all_conv_blocks):.2f}")
print(f"Total cudnn ms: {sum(conv.get_cudnn_ms() for conv in all_conv_blocks):.2f}")
