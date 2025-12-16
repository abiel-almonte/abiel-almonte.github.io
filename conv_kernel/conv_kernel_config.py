from dataclasses import dataclass, field
from math import ceil
    

RTX_5080_BANDWIDTH: int = 960 # GB/s
BYTES_PER_FLOAT: int = 4

@dataclass
class BlockDim:
    x: int = 8
    y: int = 8
    z: int = 4
    
@dataclass
class LaunchConfig:
    blockdim: BlockDim = field(default_factory=BlockDim)
    
    def __post_init__(self):
        self.threads_per_block = self.blockdim.x * self.blockdim.y * self.blockdim.z
    
@dataclass
class Res:
    h: int
    w: int

@dataclass
class TensorShape(Res):
    ch: int
    
    def __str__(self):
        return f"{self.h}x{self.w}, {self.ch} channels"

@dataclass
class Filter(Res):
    out_ch: int
    stride: int = 1
    
    def __post_init__(self):
        assert(self.h == self.w)
    
    def __str__(self):
        return f"{self.h}x{self.w}, {self.out_ch} out channels, {self.stride} stride"

    
@dataclass
class Conv:
    in_tensor: TensorShape
    filter: Filter
    out_tensor: TensorShape = None
    launch_config: LaunchConfig = field(default_factory=LaunchConfig)
    
    bs: int = 1
    pad: bool = True # force padding

    def __post_init__(self):
        padding_map = {7: 3, 3: 1}
        padding = 0
        
        if self.pad:
            padding = padding_map.get(self.filter.h, 0)
        
        out_h = 1 + (self.in_tensor.h + 2 * padding - self.filter.h) // self.filter.stride
        out_w = 1 + (self.in_tensor.w + 2 * padding - self.filter.w) // self.filter.stride
        
        self.out_tensor = TensorShape(out_h, out_w, self.filter.out_ch)
    

    def get_naive_MBs(self):
        if not hasattr(self, "naive_MBs"):
            threads_per_block = self.launch_config.threads_per_block
            blockdim = self.launch_config.blockdim
            
            out_w_iters = ceil(self.out_tensor.w / blockdim.x)
        
            writes_per_thread = out_w_iters
            reads_per_thread = (
                out_w_iters * 
                self.in_tensor.ch *
                self.filter.h *
                self.filter.w
            ) * 2
        
            self.bytes_per_thread = BYTES_PER_FLOAT * (reads_per_thread + writes_per_thread)
            self.threads_launched  = (
                ceil(self.bs / blockdim.x) *
                ceil(self.out_tensor.ch / blockdim.y) *
                ceil(self.out_tensor.h / blockdim.z)
            ) * threads_per_block

            self.naive_MBs = self.bytes_per_thread * self.threads_launched * 1e-6
        
        return self.naive_MBs
    
    def get_naive_ms(self):
        if not hasattr(self, "naive_ms"):
            self.naive_ms = self.get_naive_MBs() / RTX_5080_BANDWIDTH
        
        return self.naive_ms

    def get_cudnn_MBs(self):
        if not hasattr(self, "cudnn_MBs"):
            
            weight_bytes = (
                self.out_tensor.ch *
                self.in_tensor.ch *
                self.filter.h *
                self.filter.h
            ) * BYTES_PER_FLOAT
        
            input_bytes = (
                self.bs *
                self.in_tensor.ch *
                self.in_tensor.h *
                self.in_tensor.w
            ) * BYTES_PER_FLOAT
        
            out_bytes = (
                self.bs * 
                self.out_tensor.ch * 
                self.out_tensor.h * 
                self.out_tensor.w
            ) * BYTES_PER_FLOAT

            self.cudnn_MBs = (weight_bytes + input_bytes + out_bytes) * 1e-6
    
        return self.cudnn_MBs
    
    def get_cudnn_ms(self):
        if not hasattr(self, "cudnn_ms"):
            self.cudnn_ms = self.get_cudnn_MBs() / RTX_5080_BANDWIDTH
        
        return self.cudnn_ms
    
    
class ConvBlock:
    def __init__(
        self,
        input_tensor: TensorShape,
        filter_list: list[Filter], 
        downsample: bool = False
    ):
        if downsample:
            f = filter_list[0]
            filter_list[0] = Filter(h=f.h, w=f.w, out_ch=f.out_ch, stride=2) # create new filter
    
        self.conv_list: list[Conv] = []
        
        prev = input_tensor
        for filter in filter_list:
            
            self.conv_list.append(
                conv := Conv(prev, filter)
            )
            
            prev = conv.out_tensor
        
        self.out_tensor = prev
        
        ffilter = filter_list[0]
        lfilter = filter_list[-1]
        
        self.conv_list.append(
            Conv(
                input_tensor,
                Filter(h=1, w=1, out_ch=lfilter.out_ch, stride=ffilter.stride)
            ) # conv for residual
        )
    
    def get_naive_MBs(self):
        if not hasattr(self, "naive_MBs"):
            self.naive_MBs = sum(conv.get_naive_MBs() for conv in self.conv_list)
        
        return self.naive_MBs
    
    def get_naive_ms(self):
        if not hasattr(self, "naive_ms"):
            self.naive_ms = sum(conv.get_naive_ms() for conv in self.conv_list)
        
        return self.naive_ms
    
    def get_cudnn_MBs(self):
        if not hasattr(self, "cudnn_MBs"):
            self.cudnn_MBs = sum(conv.get_cudnn_MBs() for conv in self.conv_list)
        
        return self.cudnn_MBs
    
    def get_cudnn_ms(self):
        if not hasattr(self, "cudnn_ms"):
            self.cudnn_ms = sum(conv.get_cudnn_ms() for conv in self.conv_list)
        
        return self.cudnn_ms

    def get_stats(self) -> str:
        s = f"Total naive MBs: {self.get_naive_MBs():.2f}\n"
        s += f"Total naive ms: {self.get_naive_ms():.2f}\n"
        s += f"\nTotal cudnn MBs: {self.get_cudnn_MBs():.2f}\n"
        s += f"Total cudnn ms: {self.get_cudnn_ms():.2f}"

        return s

    def __str__(self): 
        line  = "=" * 50 + "\n"

        s = "Block:\n\n"
        for i, conv in enumerate(self.conv_list[:-1]):
            s += f"layer {i:>2}:".rjust(12) + "\t" + str(conv.out_tensor).rjust(12) + "\n"
        
        s += "\n"
        s += f"Residual:".rjust(12) + "\t" + str(self.conv_list[-1].in_tensor) + " -> " + str(self.conv_list[-1].out_tensor) + "\n"
        
        s += line
        s += self.get_stats() + "\n"
        
        return s
