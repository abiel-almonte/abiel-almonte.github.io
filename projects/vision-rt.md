---
layout: single
title: "Vision-RT: Real-time Image Classification"
permalink: /projects/vision-rt/
---

When sub-millisecond latency matters, traditional CV pipelines often fall short. Vision-RT is a minimal developer toolkit that reduces significant overhead to enable simple real-time CV tasks on Linux.

The philosophy here is creating a system sculpted for our specific use case, rather than accepting the baggage of general-purpose frameworks.

The two pain points addressed here were slow frame acquisition through OpenCV and inefficient static PyTorch inferencing, replaced with a custom V4L2 pipeline and computational graph capture respectively.

In our benchmarks, Vision-RT accelerated image classification pipeline by over 10x compared to conventional methods.

![Same workload 12x faster](/vision-rt.png)

[View on GitHub](https://github.com/Abiel-Almonte/vision-rt){: .btn .btn--primary}

## Finding the bottleneck
TODO