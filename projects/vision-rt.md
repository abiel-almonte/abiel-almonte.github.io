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

In our benchmarks, Vision-RT accelerated image classification pipeline by over **10x** compared to conventional methods.

<div class="figure">
  <img src="/vision-rt.png" alt="Same workload 12x faster">
  <p class="figure-caption">Fig. 1: VisionRT fits within the 90 FPS frame budget. The standard pipeline overruns, dropping to ~40 FPS.</p>
</div>

## Finding the bottleneck
TODO