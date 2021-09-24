---
template: lesson.html
title: Continual learning
description: Architecting a continual system that iteratively updates and gains our trust over time.
keywords: continual learning, retraining, monitoring, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Continual learning

In the pipelines lesson, we covered the [Dataops](pipelines.md#dataops){:target="_blank"}, MLOps ([model](pipelines.md#mlops-model){:target="_blank"} and [update](pipelines.md#mlops-update){:target="_blank"}) workflows needed to train and update our model. In this lesson, we'll conclude by looking at how these different workflows all connect to create an ML system that's continually learning. We use the word `continual` (repeat with breaks) instead of `continuous` (repeat without interruption / intervention) because we're **not** trying to create a system that will automatically update with new incoming data without human intervention.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/continual_learning/connect.png" width="1000" alt="continual learning system">
</div>
<div class="ai-center-all">
  <small>A simplified view to illustrate how workflows connect. Not depicting <a href="versioning.md" target="_blank">version control</a>, <a href="cicd.md" target="_blank">CI/CD</a> across multiple environments, <a href="cicd.md#deployment" target="_blank">deployment</a>/<a href="infrastructure.md#testing" target="_blank">testing</a> strategies, etc.</small>
</div>

A continual learning system like this will guide us with when to update, what exactly to update and how to update it (easily). The goal is not always to build a continuous system that automatically updates but rather a continual system that iteratively updates and gains our trust over time. Though we’ve closed the iteration loop for our continual system with our [update workflow](pipelines.md#mlops-update){:_target="blank"}, there are many decisions involved that prevent the iteration from occurring continuously.

!!! note
    Continual learning not only applies to MLOps workflow architecture but on the [algorithmic front](https://arxiv.org/abs/1909.08383){:target="_blank"} as well, where models learn to adapt to new data without having to retrain or suffer from catastrophic forgetting (forget previously learned patterns when presented with new data).

### Monitoring

Starting backwards from our update workflow, there are so many moving pieces involved with monitoring. What values to monitor for drift (data, target, concept), how to measure drift (chi^2, KS, MMD, etc.), window sizes/frequencies, thresholds for triggering alerts and more. Once an alert is actually triggered, what events take place? These policies evolve based on experience from previous iterations, domain expertise, etc. And how do we avoid alerting fatigue and actually identify the root cause (we need smart slicing & aggregation).

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pipelines/update.png" width="1000" alt="continual learning system">
</div>

### Retraining

If the appropriate action is to retrain, it’s not just the matter of fact of retraining on the old data + new data (if only it was that easy)! There’s an entire workflow (often human-in-the-loop) that is responsible for composing the retraining dataset. We use the word “compose” because it really is an art. Labeling, active learning, views for domain experts, quality assurance, augmentation, up/down sampling, evaluation dataset with appropriate slice representations, etc. Read more about the nuances of this process in our [data-driven development lesson](data-driven-development.md){:target="_blank"} which focuses on data-driven development and treating data as the first class citizen.

### Evaluation

There are decisions in the DataOps workflows that need to be reevaluated as well such as adding to the set of [expectations](testing.md#expectations){:target="_blank"} that our features and models need to pass. These evaluation criteria are added a result of our system interacting with the real world. Usually, with the exception of large concept drift, these expectations should remain valid and should require fewer updates after the first several iterations.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pipelines/dataops.png" width="1000" alt="continual learning system">
</div>

### Impact

These are just a few of the dynamic levers that need to be evaluated and adjusted after nearly every iteration. So the brunt of the work comes after shipping the first model and when trying to keep our system relevant to the changing world (developing user preferences, social trends, pandemics, etc.). So, all this work better be worth it, such that even incremental improvements translate to meaningful impact! But the good news is that we’ve created a highly transparent system that can adapt to any change with fewer and fewer technical interventions, thanks to both the increased data to learn from and the learned evaluation, monitoring and update policy criteria in place.

!!! warning
    While it's important to iterate and optimize the internals of our workflows, it's even more important to ensure that our ML system are actually making an impact. We need to constantly engage with stakeholders (management, users) to iterate on **why** our ML system exists.

### Tooling
Tooling is improving ([monitoring](monitoring.md){:target="_blank"}, [evaluation](testing.md){:target="_blank"}, [feature store](feature-store.md){:target="_blank"}, etc.) & control planes (ex. [Metaflow](https://metaflow.org/){:target="_blank"}) are emerging to connect them. Even tooling to allow domain experts to interact with the ML system beyond just labeling (dataset curation, segmented monitoring investigations, explainability). A lot of companies are also building centralized ML platforms (ex. Uber's [Michelangelo](https://eng.uber.com/michelangelo-machine-learning-platform/){:target="_blank"} or LinkedIn's [ProML](https://engineering.linkedin.com/blog/2019/01/scaling-machine-learning-productivity-at-linkedin){:target="_blank"}) to allow their developers to move faster and not create redundant workflows (shared feature stores, health assurance monitoring pipelines, etc.)

<!-- Citation -->
{% include "cite.md" %}