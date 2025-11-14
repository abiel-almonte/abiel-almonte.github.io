---
layout: splash
title: Projects
permalink: /projects/
---

<style>
  .projects-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 3rem 2rem;
  }
  .project {
    display: flex;
    gap: 3rem;
    margin: 4rem 0;
    align-items: center;
    background: rgba(0,0,0,0.1);
    padding: 2.5rem;
    border-radius: 12px;
  }
  .project-image {
    flex-shrink: 0;
    width: 400px;
    height: 280px;
    object-fit: cover;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  .project-content {
    flex: 1;
  }
  .project-content h2 {
    margin-top: 0;
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
  }
  .project-content strong {
    font-size: 1.2rem;
    display: block;
    margin-bottom: 1rem;
  }
  @media (max-width: 968px) {
    .project {
      flex-direction: column;
      gap: 1.5rem;
    }
    .project-image {
      width: 100%;
      max-width: 500px;
      height: auto;
    }
  }
</style>

<div class="projects-container">

  <div class="project">
    <img src="/vision-rt.png" alt="Vision-RT" class="project-image">
    <div class="project-content">
      <h2>Vision-RT</h2>
      <strong>Real-time image classification with ultra-low latency</strong>
      <p>Ditched OpenCV and cut end-to-end latency for image classification to a fraction of the original time through optimized pipelines.</p>
      <a href="https://github.com/Abiel-Almonte/vision-rt" class="btn btn--primary">View on GitHub</a>
    </div>
  </div>

</div>
