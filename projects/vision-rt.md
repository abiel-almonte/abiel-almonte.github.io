---
layout: splash
title: "Vision-RT: Real-time Image Classification"
permalink: /projects/vision-rt/
---

<style>
  .page__content {
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
    padding: 2rem;
    font-size: 15px;
    line-height: 1.6;
  }
  .page__content h2 {
    font-size: 1.4rem;
    margin-top: 2rem;
  }
  .page__content h3 {
    font-size: 1.15rem;
  }
  .page__content p {
    margin-bottom: 1.2rem;
  }
  .page__content img {
    display: block;
    margin: 2rem auto;
    max-width: 100%;
  }
  .page__content ul {
    margin-top: 0.25rem;
    margin-bottom: 0.5rem;
  }
  .page__content li {
    margin-bottom: 0.15rem;
  }
</style>

## Vision-RT: Real-time Image Classification

When sub-millisecond latency matters, traditional CV pipelines often fall short. Vision-RT is a minimal developer toolkit that reduces significant overhead to enable simple real-time CV tasks on Linux.

The philosophy here is creating a system sculpted for our specific use case, rather than accepting the baggage of general-purpose frameworks.

Vision-RT addresses two bottlenecks:
- Slow frame acquisition through OpenCV.
- Inefficient static PyTorch inferencing.

We replaced these with a custom V4L2 pipeline for direct camera access and CUDA graph capture for optimized model execution.

In our benchmarks, Vision-RT accelerated image classification pipeline by over 10x compared to conventional methods.

![Same workload 12x faster](/vision-rt.png)

[View on GitHub](https://github.com/Abiel-Almonte/vision-rt){: .btn .btn--primary}

## Finding the bottleneck
TODO