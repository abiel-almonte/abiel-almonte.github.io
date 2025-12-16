def naive_conv_kernel(in_tensor, filter, stride, out_tensor):
    
    # in.shape = [batch_size, in_channels, in_height, in_width]
    # filter.shape = [out_channels,in_channels, filter_height, filter_width]
    # out.shape = [batch_size, out_channels, out_height, out_width]

    bs, in_ch, in_h, in_w = in_tensor.shape
    out_ch, _, f_h, f_w = filter.shape
    
    out_h = int(1 + (in_h - f_h) / stride)
    out_w = int(1 + (in_w - f_w) / stride)
        
    for batch in range(bs):
        
        for out_channel in range(out_ch):
            for out_row in range(out_h):
                for out_col in range(out_w):
                    
                    sum = 0.0
                    
                    for in_channel in range(in_ch):
                        for filter_row in range(f_h):
                            for filter_col in range(f_w):
                        
                                filter_element = filter[out_channel][in_channel][filter_row][filter_col]
                                
                                in_row = filter_row + out_row * stride
                                in_col = filter_col + out_col * stride
                                in_element = in_tensor[batch][in_channel][in_row][in_col]
                                
                                sum += filter_element * in_element
                    
                    out_tensor[batch][out_channel][out_row][out_col] = sum
                    
    return out_tensor
