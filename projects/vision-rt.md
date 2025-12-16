---
layout: splash
title: "Vision-RT"
permalink: /projects/vision-rt/
---

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">

<style>
  .page__content {
    max-width: 1000px;
    margin-left: auto;
    margin-right: auto;
    padding: 2rem;
    font-family: 'Inter', Arial, Helvetica, sans-serif !important;
    font-size: 16px;
    line-height: 20px;
  }
  .page__content h2 {
    font-size: 32px;
    font-weight: 700;
    margin-top: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    width: 100%;
  }
  .page__content h2 small {
    font-size: 22px;
    font-weight: normal;
    margin-left: auto;
  }
  .page__content h3 {
    font-size: 16px;
    font-weight: 700;
  }
  .page__content p {
    margin-bottom: 1rem;
  }
  .page__content ul {
    margin-bottom: 1rem;
  }
  .page__content li {
    margin-bottom: 0.5rem;
  }
  .figure {
    text-align: center;
    margin: 1rem 0;
  }
  .figure img {
    max-width: 100%;
  }
  .figure img.large {
    max-width: none;
    display: block;
    position: relative;
    left: 50%;
    transform: translateX(-50%);
  }
  .figure-caption {
    font-size: 14px !important;
    color: #666;
    font-style: italic;
    margin-top: 0.5rem;
  }
  figure-caption {
    font-size: 14px !important;
    color: #666;
    font-style: italic;
    margin-top: 0.5rem;
    display: block;
  }
  .page__content table {
    font-size: 14px !important;
    margin: 1rem auto;
    display: table;
  }
  .page__content table td,
  .page__content table th {
    font-size: 14px !important;
    padding: 0.5rem;
  }
  
</style>

## Vision-RT -- Ditching OpenCV <small>[View on GitHub](https://github.com/Abiel-Almonte/vision-rt)</small>

When sub-millisecond latency matters, traditional CV pipelines often fall short. Vision-RT is a minimal developer toolkit that reduces significant overhead to enable simple real-time CV tasks on Linux.

The philosophy here is creating a system sculpted for our specific use case, rather than accepting the baggage of general-purpose frameworks.

Vision-RT addresses two bottlenecks:
- Slow frame acquisition through OpenCV
- Inefficient static PyTorch inferencing

We replaced these with a custom V4L2 pipeline for direct camera access and CUDA graph capture for optimized model execution.

In our benchmarks, Vision-RT accelerated image classification pipeline by over **2x** compared to conventional methods.


<div class="figure">
  <img src="/images/vision-rt/vision-rt2.png" alt="Zero overhead">
  <p class="figure-caption">Fig. 1: VisionRT fits within the 90 FPS frame budget. The standard pipeline overruns, dropping to ~40 FPS.</p>
</div>


With the initial goal of accelerating perception for robotics, I decided to profile first to find exactly where the time sinks are.

## Finding the Bottleneck

Throughout the rest of the blog, I will use `nsys` and `nvtx` to profile the image classification pipeline in search of potential areas for optimization.

The pipeline is split into 3 stages:
- **Capture** - Fetching the frame buffer.
- **Preprocessing** - Creating a PyTorch-compatible tensor.
- **Inference**-  ResNet50's forward propagation.

These stages are abstracted in the following profiled code:
```python
@nvtx.annotate("standard", color="blue")
def run_standard(cap, model):
    frame = capture_overhead(cap)
    if frame is False:
        return False
        
    tensor = preprocessing(frame)
    inference(tensor, model)

    return True
```

Here are the summary after profiling 10K samples post-warmup:

<div class="figure">
  <img src="/images/vision-rt/profile_stats1.png" alt="Infernce profile overview" width=2068 height=195>
  <p class="figure-caption">Fig. 2: Result of profiling with nsys and nvtx</p>
</div>

`Fig. 2` shows that `Capture` and `Inference` dominate the latency with an average of `12.2 ms` and `9.4 ms`, respectively. We also observe significant variance in both stages' latency, which can violate real-time system requirements.

Assuming average latency, lets analyze the potential cascading affect.

## Quantifying the Impact

For our baseline pipeline with negligible preprocessing overhead:

1. Mean capture latency: `12.2 ms`
2. Mean inference latency: `9.4 ms`  
3. Total pipeline latency: `21.6 ms per frame`

The camera operates at `90 Hz`, establishing a frame period of `11.11 ms`. This represents our real-time budget, the maximum processing time to maintain synchronous operation.

With `21.6 ms` actual processing time, we accumulate `10.49 ms` of delay per frame processed. This deficit compounds deterministically:

**Analysis over 100 frames:**

| Measurement | Calculation | Time |
|-----------|---------|---------|
| Processing time | `100 * 21.6 ms` | 2,160 ms |
| Real time elapsed | `100 * 11.11 ms` | 1,111 ms |
| **Accumulated deficit** | `2,160 ms - 1,111 ms` | **1,049 ms** |

This `1.05 second`deficit corresponds to `94` dropped frames (`1,049 ms // 11.11 ms`).

| Metric | Calculation | Result |
|-----------|---------|---------|
| Efficiency | `100 frames processed / 194 frames produced` | 51.4% |
| **Effective throughput** | `51.4% * 90 FPS` | **46.3 FPS** |


The baseline pipeline is therefore latency-bound by a factor of `~2x`, explaining the observed degradation from `90 FPS` to `~45 FPS` under sustained load. The system cannot operate at the camera's native frame rate.

Now we'll determine whether to target `Capture` or `Inference` first. To maxmize potential gains, we'll calculate their lower bound to see which stage has more headroom.

## Calculating the Lower Bound

**Pipeline Specifications**

| Component | Specification | Value |
|-----------|--------------|-------|
| Resolution | 320 ✕ 240 | 76,800 pixels |
| Frame Format | YUYV (4:2:2) | 2 bytes/pixel |
| Frame Buffer | 320 ✕ 240 ✕ 2 | 153.6 KB |
| Refresh Rate | 90 Hz | 11.11ms period |
| Interface | USB 2.0 | 60 MB/s bandwidth |
| GPU PCIe | RTX 5080 | 960 GB/s bandwidth |
| GPU FPU | RTX 5080 | 56.28 TFLOPS |

# Capture Lower Bound
To establish a theoretical minimum for capture, I'll trace the data flow from camera to GPU-ready tensor.

**Frame Acquisition Time**

| Operation | Calculation | Time |
|-----------|------------|------|
| Camera frame period | `1 / 90 Hz` | 11.11ms |
| USB 2.0 transfer | `153.6 KB / 60 MB/s` | 2.02ms |
| **Frame acquisition** | `max(11.11ms, 2.02ms)` | **11.11ms** |

**YUYV-2-RGB-Normalization Kernel**

| Operation | Calculation | Result |
|-----------|---------|---------|
| Threads | `153.6 KB / 4B` | 38.4K threads |
| Integer unpacking | `56 ops ✕ 38.4K threads` | *Not limiting* |
| Compute | `(12 ops ✕ 38.4K threads) / 56.28 TFLOPS` | 8.2ns |
| Memory | `(28 bytes ✕ 38.4K threads) / 960 GB/s` | 1.12µs |
| **Kernel time** | `max(compute, memory)` | **1.12μs** |

---

**Capture Theoretical Minimum**

| Stage | Time |
|-------|------|
| Frame acquisition | 11.11 ms |
| Host-to-Device copy | 0.16μs |
| Kernel time | 1.12μs |
| **Capture Lower Bound** | **~11.11ms** |

# Inference Lower Bound

To establish a theoretical minimum for inference, I'll analyze the computational requirements of ResNet50 by examining a single convolution operation and scaling to the full network.

**Analyzing Conv2d Operations**

I wrote a naive 2D convolution kernel to understand the exact operations required:

```cpp
    // block parallelize over the first three for-loops.
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_channel = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_row = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (batch >= bs || out_channel >= out_ch || out_row >= out_h) {
        return;
    }
    
    // thread parallelize over the the 4th for-loop.
    for (int out_col = threadIdx.x; out_col < out_w; out_col += blockDim.x) {
        
      auto dot_product = (T) 0;
      
      for (int in_channel = 0; in_channel < in_ch; in_channel += 1){
            for (int filter_row = 0; filter_row < f_h; filter_row += 1){
                for (int filter_col = 0; filter_col < f_w; filter_col += 1){
                    
                    const T filter_element = filter[out_channel][in_channel][filter_row][filter_col];
                    
                    const int image_row = filter_row + out_row * stride;
                    const int image_col = filter_col + out_col * stride;
                    const T image_element = image[batch][in_channel][image_row][image_col];
                    
                    dot_product += filter_element * image_element;
                }
            }
        }
        
        out[batch][out_channel][out_row][out_col] = dot_product;
    }
```

**Per-thread analysis:**  
Each output element requires approximately:

| Metric             | Formula                                                   |
|--------------------|:----------------------------------------------------------|
| Compute   | `ceil(out_w / blockdim.x) ✕ (in_ch ✕ f_h ✕ f_w) MACs`          |
| Memory | `4 ✕ ceil(out_w / blockdim.x) ✕ [2 ✕ (in_ch ✕ f_h ✕ f_w) + 1] Bytes`|

<small> *Note:* `out_w = 1 + (in_h - f_h) // stride`</small>  

The formula for the kernel's total FLOPs and MBs, assuming  all threads launched do work, become the following:

| Metric           | Formula                                                                                                                        |
|:-----------------|:-------------------------------------------------------------------------------------------------------------------------------|
| FLOPs per Thread | `2 ✕ ceil(out_w / blockdim.x) ✕ in_ch ✕ f_h ✕ f_w`                                                                             |
| MBs per Thread | `4 ✕ ceil(out_w / blockdim.x) ✕ [2 ✕ (in_ch ✕ f_h ✕ f_w) + 1] ✕ 1e-6`                                                          |
| Threads per Block| `blockdim.x ✕ blockdim.y ✕ blockdim.z`                                                                                         |
| Number of Blocks | `ceil(bs / blockdim.x) ✕ ceil(out_ch / blockdim.y) ✕ ceil(out_h / blockdim.z)`                                                 |
| **Total est. FLOPs** | `FLOPs per Thread ✕ Threads per Block ✕ Number of Blocks`                                                             |
| **Total est. MBs** | `MBs per Thread ✕ Threads per Block ✕ Number of Blocks`                                                             |

<small> *Note: Each multiply-accumulate (MAC) counts as 2 FLOPs*</small>  

Despite the verbosity, the total FLOPs can be nicely simplifed into the following well-known formula when the tensor shapes are "nice".

```
Total FLOPs = 2 ✕ bs ✕ (f_h ✕ f_w ✕ in_ch) ✕ (out_h ✕ out_w ✕ out_ch)
```

<small> *Note: `Total FLOPs <= Total est. FLOPs`*</small>

We'll use a simple example to determine whether this kernel is compute or memory bound, assuming the following launch config:

```cpp
dim3 threadGrid(8, 8, 4); // 256 threads per block
dim3 blockGrid(
    (bs + 7) / 8, // batch dimension
    (out_ch + 7) / 8, // out channel dimension
    (out_h + 3) / 4 // out height dimension
);
```
Given a `226×226` RGB image, `3×3` filter, and parameters `[stride=1, out_ch=64]`:


| Per-Thread          | Calculation                                 | Result  |
|--------------------|---------------------------------------------|---------|
| out_w iters        | `ceil(224 / 8)`                            | 28 iterations |
| MACs               | `28 × (3 × 3 × 3)`                         | 756 MACs |
| reads       | `756 × 2 elements × 4B`                    | 6,048 bytes |
| writes      | `28 elements × 4B`                         | 112 bytes |
| **FLOPs**              | `756 × 2`                                   | **1,512 FLOPs** |
| **Total memory**       | `6,048 + 112`                              | **6,160 bytes** |

| Device-Wide        | Calculation                                                      | Result  |
|--------------------|------------------------------------------------------------------|---------|
| Total Threads      | `1 batch × 8 out_ch_blocks × 56 out_h_blocks × 256 threads/block` | 114,688 threads |
| Compute            | `(1,512 ops × 114,688 threads) / 56.28 TFLOPS`                  | 3.08 µs |
| Memory             | `(6,160 bytes × 114,688 threads) / 960 GB/s`                    | 0.736 ms |
| **naive Kernel time**    | `max(compute, memory)`                                          | **0.736 ms** |

The kernel is clearly memory-bound.

However when profiling convolution directly on pytorch 
```python
@nvtx.annotate("conv_pytorch", color="black")
def conv(x, w):
    return F.conv2d(x, w, stride=[1,1])
```
 the results were shocking:

<div class="figure">
  <img src="/images/vision-rt/conv_cudnn_profile.png" alt="" width=2068 height=195>
  <p class="figure-caption">Fig. 3: Result of profiling cudnn convlution</p>
</div>

The convlution kernel averaged just `15 µs` and a minimum of `14.8µ`! Suprisingly this actually aligns with the following formula for the *optimal* kernel, where data is moved only once for input, weight, and output.

|         | Calculation                                                      | Result  |
|--------------------|------------------------------------------------------------------|---------|
| Input      | `4 ✕ bs ✕ in_ch ✕ in_h ✕ in_w` | 612912 bytes |
| Weight        | `4 ✕ f_h ✕ f_w ✕ in_ch ✕ out_ch`                  | 6912 bytes |
| Output         | `4 ✕ bs ✕ out_ch ✕ out_h ✕ out_w`                    | 12845056 bytes |
| **cudnn Kernel time**    | `(Input + Weight + Output) / 960 GB/s`                               | **0.14 µs** |

Knowing this we can extrapolate the same formula to the entire model to find the lower bound, with the assumption that memory bandwidth is the limiting factor.

Below is the ResNet-50 architecture:

<div class="figure">
  <img src="/images/vision-rt/resnet50_flops.png" alt="ResNet Layers" >
  <p class="figure-caption">Fig. 3: ResNet model architecture, taken from "Deep Residual Learning for Image Recognition" by Kaiming He et al. (2015).</p>
</div>

| Stage  | Calculation  | Latency |
|--------------------|----|
| conv1| `7.67 MB / 960 GB/s`| 0.01 ms|
| conv2| `31.36 MB / 960 GB/s`| 0.03 ms |
| conv3| `30.54 MB / 960 GB/s`| 0.03 ms |
| conv4| `45.97 MB / 960 GB/s`| 0.05 ms |
| conv5| `64.99 MB / 960 GB/s`| 0.07 ms |
| fc | `8.2 MB / 960 GB/s` | 0.01 ms|
| **Inference lower bound**| `conv1 + … + conv5 + fc`| **0.2 ms**                             

---

|Stage | Lower Bound | Headroom | Possible Reduction in Latency (%)|
|-|
| Capture | 11.11ms | 1.126ms | 9%|
| **Inference** | 0.2ms | 9.167ms | **98%**|

So knowing that `Inference` has a significantly more headroom than `Capture`. Let's take a closer look into `Inference` …

## Optimizing Inference

```python
@nvtx.annotate("inference", color="red")
def inference(tensor, model):
    _ = model(tensor)
    torch.cuda.synchronize()
```

Here we zoom into a single `Inference` sample on the profiler:

<div class="figure">
  <img src="/images/vision-rt/inference_profile1.png" alt="Inference profile overview" width=734 height=512>
  <p class="figure-caption">Fig. 4: An annotated view of ResNet50 Inference on nsys's profiler</p>
</div>

`Fig 4` is how the profiler looks for every GPU kernel executed. For each kernel the following work must be done. 

1. **CPU work** – Framework overhead and CPU computation before launch.
2. **Kernel Launch** – CPU enqueues kernel into CUDA stream.
3. **GPU Scheduling** – GPU schedules kernel when resources are met.
4. **Kernel Execution** – GPU performs computation asynchronously.
5. **Synchronization** (if needed) – Host blocks on CUDA sync or blocking API call.

We can minimize the overhead and jitter surrounding kernel execution by capturing the [CUDA graph](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) and replaying it each iteration with a single kernel launch as long as the shapes and computation remain static. 

Here we record the computational graph of the forward function `fn` from PyTorch's default CUDA stream.

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
{
    c10::cuda::CUDAStream capture_stream = c10::cuda::getStreamFromExternal(stream, 0);
    c10::cuda::CUDAStreamGuard guard(capture_stream);
    
    out.copy_(fn(in).cast<torch::Tensor>());
}
cudaStreamEndCapture(stream, &graph);
```

 Ideally, this will eliminate the white space inbetween each `Kernel Execution` in `Fig 4`, for example we illustrate this idea below:

<div class="figure">
  <img src="/images/vision-rt/graph2.png" alt="CUDA graph drawing" class="large" width="1100">
  <p class="figure-caption">Fig. 5: Conceptual illustration to show the benefit of CUDA graphs.</p>
</div>

`Communication` refers to the overhead/ latency of CPU-GPU coordination. `Fig. 5` demonstrates how graph capture and replay eliminates communication between each operation, significantly reducing end-to-end latency.

The table below highlights the significant drop in inference times achieved with this optimization:

<div class="figure">
  <img src="/images/vision-rt/profile_stats2.png" alt="Infernce profile overview" width=2068 height=195>
  <p class="figure-caption">Fig. 6: Result of profiling post CUDA graph optimization with nsys and nvtx</p>
</div>

With CUDA graphs, the average inference latency drops from `9.4ms` to just `1.35ms`. Inference times are also much more predictable, the standard deviation has decreased dramatically from `2.8ms` to just `81µs`!

Now that inference is no longer a bottleneck lets tackle capture overhead.

## Optimizing Capture Overhead
  
Let's revisit the profile summary used to determine the bottlenecks shown in `Fig. 2`.

<div class="figure">
  <img src="/images/vision-rt/profile_stats3.png" alt="Infernce profile overview" width=2068 height=195>
  <p class="figure-caption">Fig. 7: Result of profiling the baseline with nsys and nvtx</p>
</div>

TODO

## Surprisingly Deterministic
<div class="figure">
  <img src="/images/vision-rt/latency_kde.png" alt="Deterministic Latency">
  <p class="figure-caption">Fig. last: VisionRT achieves deterministic sub-12ms latency while OpenCV varies unpredictably from 20-30ms</p>
</div>

TODO