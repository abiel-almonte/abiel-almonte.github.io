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
  .figure-caption {
    font-size: 14px !important;
    color: #666;
    font-style: italic;
    margin-top: 0.5rem;
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
  <img src="/images/vision-rt/vision-rt.png" alt="Zero overhead">
  <p class="figure-caption">Fig. 1: VisionRT fits within the 90 FPS frame budget. The standard pipeline overruns, dropping to ~40 FPS.</p>
</div>


With the initial goal of accelerating percseption for robotics, I decided to profile first to find exactly where the time sinks are.

## Finding the Bottleneck

Throughout the rest of the blog, I will use `nsys` and `nvtx` to profile the image classification pipeline in search of potential areas for optimization.

The pipeline is split into 3 stages:
- **Capture** - Fetching the frame buffer.
- **Preprocessing** - Creating a PyTorch-compatible tensor.
- **Inference**-  ResNet50's forward propagation.

These stages are abstracted in the following code:
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

After profiling 100 samples post-warmup, `nsys` shows that `Capture` and `Inference` dominate the latency with an average of 11.9 ms and 9.5 ms respectivley.

<div class="figure">
  <img src="/images/vision-rt/profile_stats1.png" alt="Infernce profile overview">
  <p class="figure-caption">Fig. 2: Result of profiling with nsys and nvtx </p>
</div>

In `Fig. 2` we also observe that both stages have a large standard deviation which is concerning for systems that require real-time guarantees.

For `inference`, this variance can be attributed to the non-deterministic nature of GPU scheduling, where the GPU's hardware schedulers dynamically assign threads and warps to execution units based on resource availability. This dynamic behavior introduces jitter.

Let's take a closer look into `inference` for ResNet50 ...

```python
@nvtx.annotate("inference", color="red")
def inference(tensor, model):
    _ = model(tensor)
    torch.cuda.synchronize()
```

Here we zoom into a single `Inference` sample on the profiler:

<div class="figure">
  <img src="/images/vision-rt/inference_profile1.png" alt="Infernce profile overview">
  <p class="figure-caption">Fig. 3: An annotated view of ResNet50 inference on nsys's profiler</p>
</div>

`Fig 3` shows how the profiler looks for every GPU kernel executed. For each kernel the follow work must be done. 

1. **CPU work** – Framework overhead and CPU computation before launch.
2. **Kernel Launch** – CPU enqueues kernel into CUDA stream.
3. **GPU Scheduling** – GPU schedules kernel when resources are met.
4. **Kernel Execution** – GPU performs computation asynchronously.
5. **Synchronization** (if needed) – Host blocks on CUDA sync or blocking API call.

We can minimize the overhead and jitter surrounding kernel execution by capturing the CUDA graph and replaying it each iteration, as long as the shapes and computation remain static.
 
TODO: show snippet for cuda graph capture. and then hand made visual for a before and after graph capture. then introduce what torch.compile can do. show mlir before and after. then show code snippet. after results of the inference.
    
## Surprisingly Deterministic
<div class="figure">
  <img src="/images/vision-rt/latency_kde.png" alt="Deterministic Latency">
  <p class="figure-caption">Fig. 3: VisionRT achieves deterministic sub-12ms latency while OpenCV varies unpredictably from 20-30ms</p>
</div>