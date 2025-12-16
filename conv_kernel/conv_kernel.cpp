#include <torch/torch>

template<typename T>
__global__ void conv_kernel_nopad(
    const torch::PackedAcessor<T, 4> in, // [batch_size, in_channels, in_height, in_width]
    const torch::PackedAcessor<T, 4> filter, // [out_channels,in_channels, filter_height, filter_width]
    const int bs,
    const int out_ch,
    const int out_h,
    const int stride,
    const int out_w,
    const int in_ch,
    const int f_h,
    const int f_w,
    torch::PackedAcessor<T, 4> out, // [batch_size, out_channels, out_height, out_width]
){
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_channel = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_row = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (batch >= bs || out_channel >= out_ch || out_row <= out_h) {
        return;
    }
    
    for (int out_col = threadIdx.x; out_col < out_w; out_col += blockDim.x) {
        
        T dot_product = (T) 0;
        
        for (int in_channel = 0; in_channel < in_ch; in_channel += 1){
            for (int filter_row = 0; filter_row < f_h; filter_row += 1){
                for (int filter_col = 0; filter_col < f_w; filter_col += 1){
                    
                    const T filter_element = filter[out_channel][in_channel][filter_row][filter_col];
                    
                    const int in_row = filter_row + out_row * stride;
                    const int in_col = filter_col + out_col * stride;
                    const T in_element = in[batch][in_channel][in_row][in_col];
                    
                    dot_product += filter_element * in_element;
                }
            }
        }
        
        out[batch][out_channel][out_row][out_col] = dot_product;
    }
}


torch::Tensor lauch_nopad_conv(
    const torch::Tensor in,  // [batch_size, in_channels, in_height, in_width]
    const torch::Tensor filter, // [out_channels, in_channels, filter_height, filter_width]
    const int stride
){
    
    TORCH_CHECK(in.dim() == 4, "Ensure the image has 4 dimensions. Got:" in.dim()));
    
    const int bs = in.size(0);
    const int in_ch = in.size(1);
    const int in_h = in.size(2);
    const int in_w = in.size(3);
    
    const int out_ch = filter.size(0);
    const int f_h = filter.size(2);
    const int f_w = filter.size(3);
    
    const int out_h = 1 + (in_h - f_h) / stride;
    const int out_w = 1 + (in_w - f_w) / stride;
    
    auto opts = in.options();
    torch.Tensor out = torch::zeros({bs, out_ch, out_h, out_w}, opts);
    
    dim3 threadGrid(8, 8, 4);
    dim3 blockGrid(
        (bs + 7) / 8,
        (out_ch + 7) / 8,
        (out_h + 3) / 4
    );
    
    conv_kernel_nopad<float><<<blockGrid, threadGrid>>>(
        in.PackedAcessor<torch.RestrictPtr<const float>, 4>,
        filter.PackedAcessor<torch.RestrictPtr<const float>, 4>,
        bs,
        out_ch,
        out_h,
        stride,
        out_w,
        in_ch,
        f_h,
        f_w,
        out.PackedAcessor<torch.RestrictPtr<float>, 4>
    );
    
    return out;
}