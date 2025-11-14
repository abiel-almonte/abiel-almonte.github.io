---
layout: splash
title: Projects
permalink: /projects/
---

<style>
  .projects-container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
  }
  
  .project-table {
    width: 100%;
    border: 0;
    border-spacing: 0;
    border-collapse: separate;
    margin: 2rem 0;
  }
  
  .project-table td {
    padding: 20px;
    vertical-align: middle;
  }
  
  .project-image-cell {
    width: 30%;
  }
  
  .project-content-cell {
    width: 70%;
  }
  
  .project-image {
    width: 100%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }
  
  .project-title {
    font-family: 'Lato', Verdana, Helvetica, sans-serif;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 10px;
  }
  
  .skills {
    margin: 8px 0;
  }
  
  .skill {
    background: rgba(0,0,0,0.2);
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 13px;
    margin-right: 6px;
    display: inline-block;
  }
  
  .project-links {
    margin: 10px 0;
  }
  
  .project-links a {
    margin-right: 15px;
    font-weight: 600;
  }
  
  @media (max-width: 768px) {
    .project-table td {
      display: block;
      width: 100% !important;
      padding: 10px;
    }
    .project-image-cell {
      text-align: center;
    }
  }
</style>

<div class="projects-container">

<table class="project-table">
  <tbody>
    <tr>
      <td class="project-image-cell">
        <img src="/vision-rt.png" alt="Vision-RT" class="project-image">
      </td>
      <td class="project-content-cell">
        <p class="project-title">Vision-RT</p>
        <div class="skills">
          <span class="skill">Computer Vision</span>
          <span class="skill">Real-time Systems</span>
          <span class="skill">CUDA</span>
          <span class="skill">PyTorch</span>
          <span class="skill">V4L2</span>
        </div>
        <p>
          Real-time image classification with ultra-low latency by ditching OpenCV and building optimized pipeline from scratch. Cut end-to-end latency to a fraction of the original time.
        </p>
        <div class="project-links">
          [<a href="/projects/vision-rt">deepdive blog</a>]
          [<a href="https://github.com/Abiel-Almonte/vision-rt">github</a>]
        </div>
      </td>
    </tr>
  </tbody>
</table>

<hr style="margin: 3rem 0; opacity: 0.3;">


</div>
