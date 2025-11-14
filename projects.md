---
layout: single
title: Projects
permalink: /projects/
---

<style>
  .project {
    display: flex;
    gap: 2rem;
    margin: 3rem 0;
    align-items: flex-start;
  }
  .project-image {
    flex-shrink: 0;
    width: 250px;
    height: 200px;
    object-fit: cover;
    border-radius: 8px;
  }
  .project-content {
    flex: 1;
  }
  @media (max-width: 768px) {
    .project {
      flex-direction: column;
    }
    .project-image {
      width: 100%;
    }
  }
</style>

<div class="project">
  <img src="vision-rt.png" alt="Vision-RT" class="project-image">
  <div class="project-content">
    <h2>Vision-RT</h2>
    <strong>Real-time image classification with ultra-low latency</strong>
    <p>Ditched OpenCV and cut end-to-end latency for image classification to a fraction of the original time through optimized pipelines.</p>
    <a href="https://github.com/Abiel-Almonte/vision-rt" class="btn btn--primary">View on GitHub</a>
  </div>
</div>

---
