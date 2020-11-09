---
layout: page
title: Iteration
description: Improving on our solution iteratively over time.
image: /assets/images/courses/applied-ml-in-production/iteration.png
tags: product product-management evaluation

course-url: /courses/applied-ml-in-production/
next-lesson-url: /courses/applied-ml-in-production/annotation/
---

<!-- Header -->
<div class="row">
  <div class="col-md-8 col-6 mr-auto">
    <h1 class="page-title">{{ page.title }}</h1>
  </div>
  <div class="col-md-4 col-6">
    <div class="btn-group float-right mb-0" role="group">
      <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i
          class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
    </div>
  </div>
</div>
<hr class="mt-0">

<!-- Video -->
<div class="ai-center-all mt-2">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/8ntrWE12HNE?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div>
<div class="ai-center-all mt-2">
  <small>Accompanying video for this lesson. <a href="https://www.youtube.com/madewithml?sub_confirmation=1" target="_blank">Subscribe</a> for updates!</small>
</div>


<h3><u>Intuition</u></h3>
We don't want to spend months of time developing a complicated solution only to learn that the entire problem has changed. The main idea here is to **close the loop**, which invovles:
1. Create a minimum viable product (MVP) that satisfies a baseline performance.
2. Iterate on your solution by using the feedback.
3. Constantly reassess to ensure your objective hasn't changed.

<figure>
  <img src="/assets/images/courses/applied-ml-in-production/development_cycle.png" width="700" alt="product development cycle">
  <figcaption>Product development cycle</figcaption>
</figure>

Creating the MVP for solutions that requires machine learning often involves going **manual before ML**.
- deterministic, high interpretability, low-complexity MVP (ex. rule based)
- establish baselines for objective comparisons
- allows you to ship quickly and get feedback from users

Deploying solutions is actually quite easy (from an engineering POV) but maintaining and iterating upon it is quite the challenge.
- collect signals from UX
- monitor (performance, concept drift, etc.) to know when to update
- constantly reassess your objective
- iteration bottlenecks (ex. data quality checks)

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch **[1:02 - 4:36]()** for a video walkthrough of this section.

<h3><u>Application</u></h3>
For our solution, we will have an initial set of baselines where we'll start with a rule-based approach and then slowly add complexity (regression &rarr; CNN &rarr; Transformers).
> For the purpose of this course, even our MVP will be an ML model, however we would normally deploy the rule-based approach first as long as it satisfies a performance threshold.

As for monitoring and iterating on our solution, we'll be looking at things like overall performance, class specific performances, # of relevant tags, etc. We will also create workflows to look at new data for anomalies, apply active learning, ease the annotation process, etc.

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch **[1:02 - 4:36]()** for a video walkthrough of this section.

<h3><u>Resources</u></h3>
- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design/blob/master/build/build1/consolidated.pdf){:target="_blank"}
- [Building Machine Learning Products: A Problem Well-Defined](http://jeremyjordan.me/ml-requirements/){:target="_blank"}

<!-- Footer -->
<hr>
<div class="row mb-4">
  <div class="col-6 mr-auto">
    <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
  </div>
  <div class="col-6">
    <div class="float-right">
      <a href="{{ page.next-lesson-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-right mr-1"></i>Next lesson</a>
    </div>
  </div>
</div>