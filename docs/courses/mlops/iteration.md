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

## Challenges

Deploying solutions is actually quite easy (from an engineering standpoint) but maintaining and iterating is the challenge. This is why we need to close the loop quickly first, so we can properly experiment with a process. This allows to change one thing as a time (improve data, models, etc.), evaluate it and adapt as necessary.

- collect signals from UI/UX to best [approximate performance](monitoring.md#performance){:target="_blank"}
- determine window / [rolling](monitoring.md#performance){:target="_blank"} performances on overall and key [slices](evaluation.md#slices){:target="_blank"} of data
- [monitor](monitoring.md){:target="_blank"} (performance, concept drift, etc.) to know when to update
- identify subsets of data that are worth labeling / oversampling
- address iteration bottlenecks (ex. data quality checks)
- constantly reassess your objective

> Read more about the maintenance and iteration challenges in our [data-driven development lesson](data-driven-development.md){:target="_blank"}.

## Real-world

While it may be easy (and justified) to say "don't use machine learning" as your first attempt at a solution, your leadership may not like that. Most likely, you're being paid to work on problems where traditional software isn't able to deliver.

!!! question "Justifying simple solutions"

    How can we justify the time for implementing simple, rule-based, solutions as the first solution to leadership?

    ??? quote "Show answer"
        - helps us understand the problem and discover what you don't know
        - helps us quickly identify shortcomings in the data, system, etc.
        - ease of experimentation: once we set up our system, we can easily evaluate future solutions
        - allows for conversations with domain experts for feature engineering
        - solution allows for v1 which can start the data collection process required for more sophisticated solutions

## Application
For our solution, we'll have an initial set of baselines where we'll start with a rule-based approach and then slowly add complexity (rule-based &rarr; regression &rarr; CNN). Though you are welcome to use any stochastic model you like because all the subsequent lessons are model agnostic.

!!! note
    For the purpose of this course, even our MVP will be an ML model, however we would normally deploy the rule-based approach first as long as it satisfies a performance threshold so we can quickly close the loop.

As for monitoring and iterating on our solution, we'll be looking at things like overall performance, class specific performances, # of relevant tags, etc. We'll also create workflows to look at new data for anomalies, apply active learning, ease the annotation process, etc. More on this in our [testing](testing.md){:target="_blank"} and [monitoring](monitoring.md){:target="_blank"} lessons.


## Resources
- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design/blob/master/build/build1/consolidated.pdf){:target="_blank"}
- [Building Machine Learning Products: A Problem Well-Defined](http://jeremyjordan.me/ml-requirements/){:target="_blank"}


<!-- Citation -->
{% include "cite.md" %}