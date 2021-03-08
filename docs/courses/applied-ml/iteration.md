---
template: lesson.html
title: Iteratively Improving ML Systems
description: Improving on our solution iteratively over time.
keywords: iteration, active learning, monitoring, applied ml, mlops, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/applied-ml
video: https://www.youtube.com/watch?v=Bit1IUVWrkY
---

<!-- <div class="ai-center-all mt-2">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/Bit1IUVWrkY?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div> -->

## Intuition

We don't want to spend months of time developing a complicated solution only to learn that the entire problem has changed. The main idea here is to **close the loop**, which involves:

1. Create a minimum viable product (MVP) that satisfies a baseline performance.
2. Iterate on your solution by using the feedback.
3. Constantly reassess to ensure your objective hasn't changed.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/iteration/development_cycle.png" width="700" alt="product development cycle">
</div>
<div class="ai-center-all mb-3">
  <small>Product development cycle</small>
</div>

Creating the MVP for solutions that requires machine learning often involves going **manual before ML**.

- deterministic, high interpretability, low-complexity MVP (ex. rule based)
- establish baselines for objective comparisons
- allows you to ship quickly and get feedback from users

Deploying solutions is actually quite easy (from an engineering POV) but maintaining and iterating upon it is quite the challenge.

- collect signals from UI/UX to best approximate how your deployed model is performing
- determine window / rolling performances on overall and key slices of data
- monitor (performance, concept drift, etc.) to know when to update
- constantly reassess your objective
- iteration bottlenecks (ex. data quality checks)

## Application
For our solution, we'll have an initial set of baselines where we'll start with a rule-based approach and then slowly add complexity (regression &rarr; CNN &rarr; Transformers).

!!! note
    For the purpose of this course, even our MVP will be an ML model, however we would normally deploy the rule-based approach first as long as it satisfies a performance threshold.

As for monitoring and iterating on our solution, we'll be looking at things like overall performance, class specific performances, # of relevant tags, etc. We'll also create workflows to look at new data for anomalies, apply active learning, ease the annotation process, etc.

## Resources
- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design/blob/master/build/build1/consolidated.pdf){:target="_blank"}
- [Building Machine Learning Products: A Problem Well-Defined](http://jeremyjordan.me/ml-requirements/){:target="_blank"}


<!-- Citation -->
{% include "cite.md" %}