---
layout: page
permalink: /projects/
show_title: true
wide: true
---

<table class="project-table">
  <tbody>
    <tr>
      <td class="project-image-cell">
        <img src="https://github.com/abiel-almonte/flash-recon/raw/master/images/placholder.png" alt="" class="project-image">
      </td>
      <td class="project-content-cell">
      <p class="project-title">Flash-Recon — Real-Time Monocular SLAM <span style="font-size: 14px; color: #777;">(in progress)</span></p>
      <div class="skills">
        <span class="skill">Visual SLAM</span>
        <span class="skill">CUDA</span>
        <span class="skill">PyTorch</span>
        <span class="skill">3D Reconstruction</span>
      </div>
      <p>
          Monocular SLAM system built on DROID-SLAM with fused CUDA kernels for bundle adjustment, achieving 2.5cm ATE and 10ms median latency. Integrates DepthAnythingV2 and Gaussian Splatting on a single GPU. Surfaced performance problems that motivated torq and pperf.
      </p>
      <div class="project-links">
        [<a href="https://github.com/abiel-almonte/flash-recon">code</a>]
      </div>
      </td>
    </tr>
    <tr>
      <td class="project-image-cell">
        <img src="https://github.com/abiel-almonte/torq/raw/master/images/diagram.png" alt="torq architecture" class="project-image" onerror="this.style.display='none'">
      </td>
      <td class="project-content-cell">
        <p class="project-title">Torq — Graph Compiler with CUDA-Level Interception <span style="font-size: 14px; color: #777;">(in progress)</span></p>
        <div class="skills">
          <span class="skill">CUDA</span>
          <span class="skill">C</span>
          <span class="skill">Python</span>
          <span class="skill">Compilers</span>
          <span class="skill">Quantization</span>
        </div>
        <p>
          Driver-level CUDA API interception via LD_PRELOAD for automatic graph capture, stream management, and contention detection. Building toward contention-aware dual-graph dispatch with static INT8 quantization.
        </p>
        <div class="project-links">
          [<a href="https://github.com/abiel-almonte/torq">code</a>]
        </div>
      </td>
    </tr>
    <tr>
      <td class="project-image-cell">
        <img src="https://github.com/abiel-almonte/pperf/raw/master/images/output.png" alt="pperf output" class="project-image" onerror="this.style.display='none'">
      </td>
      <td class="project-content-cell">
        <p class="project-title">pperf — Hierarchical Profiler for Quick Iteration</p>
        <div class="skills">
          <span class="skill">Python</span>
          <span class="skill">Profiling</span>
        </div>
        <p>
          Tiny hierarchical profiler that surfaces the worst bottlenecks with their full call tree. Pluggable metrics like latency, GPU memory, or anything you define. Built to profile flash-recon under real contention.
        </p>
        <div class="project-links">
          [<a href="/projects/pperf">blog post</a>]
          [<a href="https://github.com/abiel-almonte/pperf">code</a>]
        </div>
      </td>
    </tr>
    <tr>
      <td class="project-image-cell">
        <img src="https://github.com/abiel-almonte/visionrt/raw/master/images/latency_histogram.png" alt="visionrt latency histogram" class="project-image">
      </td>
      <td class="project-content-cell">
        <p class="project-title">VisionRT — Deterministic Inference via Vertical Optimization</p>
        <div class="skills">
          <span class="skill">Computer Vision</span>
          <span class="skill">Real-time Systems</span>
          <span class="skill">CUDA</span>
          <span class="skill">Triton</span>
          <span class="skill">PyTorch</span>
          <span class="skill">V4L2</span>
        </div>
        <p>
          Real-time computer vision with deterministic performance. Direct V4L2 integration, custom PyTorch compiler backend, and CUDA graph capture achieving microsecond-level timing precision.
        </p>
        <div class="project-links">
          [<a href="/projects/visionrt">blog post</a>]
          [<a href="https://github.com/abiel-almonte/visionrt">code</a>]
        </div>
      </td>
    </tr>
    <tr>
      <td class="project-image-cell">
        <img src="https://github.com/abiel-almonte/inclusive-scan/raw/main/images/sol_barchart.png" alt="inclusive-scan benchmark" class="project-image">
      </td>
      <td class="project-content-cell">
        <p class="project-title">Inclusive Scan — GPU Prefix Sum That Beats CUB</p>
        <div class="skills">
          <span class="skill">CUDA</span>
          <span class="skill">Parallel Algorithms</span>
        </div>
        <p>
          High-performance GPU prefix sum achieving 94% of theoretical DRAM bandwidth. Uses Kogge-Stone scans and decoupled lookback for cross-CTA communication. Beats NVIDIA CUB at mid-range sizes.
        </p>
        <div class="project-links">
          [<a href="https://github.com/abiel-almonte/inclusive-scan">code</a>]
        </div>
      </td>
    </tr>
    <tr>
      <td class="project-image-cell">
        <img src="https://github.com/abiel-almonte/transpose-scale/raw/master/images/bandwidth_barchart_po2.png" alt="transpose-scale benchmark" class="project-image">
      </td>
      <td class="project-content-cell">
        <p class="project-title">Transpose Scale — 3-6x Faster Than Intel MKL</p>
        <div class="skills">
          <span class="skill">C++</span>
          <span class="skill">SIMD</span>
          <span class="skill">Performance Engineering</span>
        </div>
        <p>
          High-performance matrix transpose using double cache blocking, branch elimination, and vectorized in-register operations. Cross-platform SIMD support with AVX2, SSE, and NEON.
        </p>
        <div class="project-links">
          [<a href="https://github.com/abiel-almonte/transpose-scale">code</a>]
        </div>
      </td>
    </tr>
  </tbody>
</table>
