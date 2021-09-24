---
template: lesson.html
title: Iteratively Improving ML Systems
description: Improving on our solution iteratively over time.
keywords: iteration, active learning, monitoring, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
video: https://www.youtube.com/watch?v=Bit1IUVWrkY
---

<!-- <div class="ai-center-all mt-2">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/Bit1IUVWrkY?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div> -->

{% include "styles/lesson.md" %}

## Intuition

We don't want to spend months of time developing a complicated solution only to learn that the entire problem has changed. The main idea here is to **close the loop quickly**, which involves:

1. Create a minimum viable product (MVP) that satisfies a baseline performance.
2. Iterate on your solution by using the feedback.
3. Constantly reassess to ensure your objective hasn't changed.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/iteration/development_cycle.png" width="700" alt="product development cycle">
</div>
<div class="ai-center-all mb-3">
  <small>Product development cycle</small>
</div>

Creating the MVP for solutions that requires machine learning often involves going **manual before ML**.

- deterministic, high interpretability, low-complexity MVP (ex. rule based)
- establish baselines for objective comparisons
- allows you to ship quickly and get feedback from users

!!! question "Manual solutions"

    What are examples of manual solutions?

    ??? quote "Show answer"
        - visualizations
        - aggregate analytics
        - rule-based (feature based heuristics, regex, etc.)

Deploying solutions is actually quite easy (from an engineering POV) but maintaining and iterating is the challenge.

- collect signals from UI/UX to best [approximate performance](monitoring.md#performance){:target="_blank"}
- determine window / [rolling](monitoring.md#performance){:target="_blank"} performances on overall and key [slices](evaluation.md#slices){:target="_blank"} of data
- [monitor](monitoring.md){:target="_blank"} (performance, concept drift, etc.) to know when to update
- identify subsets of data that are worth labeling / oversampling
- address iteration bottlenecks (ex. data quality checks)
- constantly reassess your objective

> Read more about the maintenance and iteration challenges in our [data-driven development lesson](data-driven-development.md){:target="_blank"}.

## Application
For our solution, we'll have an initial set of baselines where we'll start with a rule-based approach and then slowly add complexity (rule-based &rarr; regression &rarr; CNN). Though you are welcome to use any stochastic model you like because all the subsequent lessons are model agnostic.

!!! note
    For the purpose of this course, even our MVP will be an ML model, however we would normally deploy the rule-based approach first as long as it satisfies a performance threshold so we can quickly close the loop.

As for monitoring and iterating on our solution, we'll be looking at things like overall performance, class specific performances, # of relevant tags, etc. We'll also create workflows to look at new data for anomalies, apply active learning, ease the annotation process, etc. More on this in our [testing](testing.md){:target="_blank"} and [monitoring](monitoring.md){:target="_blank"} lessons.

## Real-world

While it can be humorous and easy (yet justified) to say "don't use machine learning" as your first attempt at a solution, your leadership may not find it so funny. Most likely, you're being paid a lot of money to work on problems where even complex heuristics just don't cut it.

!!! question "Justifying simple solutions"

    How can we justify the time for implementing simple, rule-based, solutions as the first solution to leadership?

    ??? quote "Show answer"
        - helps us understand the problem and discover what you don't know
        - helps us quickly identify shortcomings in the data, system, etc.
        - ease of experimentation: once we set up our system, we can easily evaluate future solutions

## Resources
- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design/blob/master/build/build1/consolidated.pdf){:target="_blank"}
- [Building Machine Learning Products: A Problem Well-Defined](http://jeremyjordan.me/ml-requirements/){:target="_blank"}


<!-- Citation -->
{% include "cite.md" %}