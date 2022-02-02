---
template: lesson.html
title: Data-centric AI
description: Leveraging data-centric views to architect a continual system that delivers trust and enables iteration.
keywords: data-centric AI, continual learning, retraining, monitoring, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

!!! warning
    This lesson is a work-in-progress, so please revisit this page later for a first draft release.

## Data-centric AI

In the previous lesson, we covered the [Dataops](pipelines.md#dataops){:target="_blank"}, MLOps ([model](pipelines.md#mlops-model){:target="_blank"} and [update](pipelines.md#mlops-update){:target="_blank"}) workflows needed to train and update our model. In this lesson, we'll conclude by looking at how these different workflows all connect to create an ML system that's continually learning while leveraging data-centric views.

> We use the word `continual` (repeat with breaks) instead of `continuous` (repeat without interruption / intervention) because we're **not** trying to create a system that will automatically update with new incoming data without human intervention.

But before we can talk about a continual learning system, we first need to discuss the key data-centric views that will enable us to make decisions along the way. We've covered most of these views extensively in previous lectures but now we'll see them in relation to the rest of the workflows for on continual learning system.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/data_centric_ai/views.png" width="500" alt="data-centric views">
</div>

These data-centric views all involve data as the first-class citizen, where developers and subject matter experts alike can interact at the data level to make decisions. And with them, we can develop a continual learning system that will guide us on when to update and what exactly to update.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/data_centric_ai/workflows.png" width="1000" alt="mlops workflows">
</div>
<div class="ai-center-all mt-3">
  <small>A simplified view to illustrate decoupled DataOps and MLOps workflows with data-centric views.</small>
</div>

> Diagram above does not depict <a href="versioning.md" target="_blank">version control</a>, <a href="cicd.md" target="_blank">CI/CD</a> across multiple environments, <a href="cicd.md#deployment" target="_blank">deployment</a>/<a href="infrastructure.md#testing" target="_blank">testing</a> strategies, etc.

The goal is not always to build a continuous system that automatically updates but rather a continual system that iteratively updates and gains our trust over time. Though we’ve closed the iteration loop for our continual system with our [update workflow](pipelines.md#mlops-update){:_target="blank"}, there are many decisions involved that prevent the iteration from occurring continuously.

!!! note
    Continual learning not only applies to MLOps workflow architecture but on the [algorithmic front](https://arxiv.org/abs/1909.08383){:target="_blank"} as well, where models learn to adapt to new data without having to retrain or suffer from catastrophic forgetting (forget previously learned patterns when presented with new data).

## Evaluation

There are decisions in our workflows that need to be [evaluated](evaluation.md){:target="_blank"} as well such as adding to the set of [expectations](testing.md#expectations){:target="_blank"} that our features and models need to pass. These evaluation criteria are added a result of our system interacting with the real world. Usually, with the exception of large concept drift, these expectations should remain valid and should require fewer updates after the first several iterations.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pipelines/dataops.png" width="1000" alt="continual learning system">
</div>

## Monitoring

There are so many moving pieces involved with [monitoring](monitoring.md){:target="_blank"}. What values to monitor for drift (data, target, concept), how to measure drift (chi^2, KS, MMD, etc.), window sizes/frequencies, thresholds for triggering alerts and more. Once an alert is actually triggered, what events take place? These policies evolve based on experience from previous iterations, domain expertise, etc. And how do we avoid alerting fatigue and actually identify the root cause (we need smart slicing & aggregation).

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pipelines/update.png" width="1000" alt="continual learning system">
</div>

## Iteration

If the appropriate action for iteration is to retrain, it’s not just the matter of fact of retraining on the old data + new data (if only it was that easy)! There’s an entire workflow (often human-in-the-loop) that is responsible for composing the retraining dataset. We use the word “compose” because it really is an art. Labeling, active learning, views for domain experts, quality assurance, augmentation, up/down sampling, evaluation dataset with appropriate slice representations, etc. Once we’ve labeled any new data and validated our new dataset, we’re ready to retrain our system. But we want to retrain in such a manner that the model learns to behave more robustly on the slices of data it previously performed poorly on.

### Upsampling

With the appropriate dataset creation workflows with labeling, QA, etc. we should have a quality updated dataset to tran our model with. However, our model will still get some things wrong. We want to identify these subsets and increase their exposure via upsampling. A recent technique proposed for this is [Just Train Twice (JTT)](https://arxiv.org/abs/2107.09044){:target="_blank"}, which initially trains a model and upsamples the training data points that the model continues to misclassify. This sampled dataset is now used to train a second model to improve performance on the “worst-group” subsets.

### Not a panacea

While retraining is a natural process of iteration, not all issues will be resolved by just retraining. Data-driven changes such as improving labeling consistency, updating labeling instructions, removing noisy samples, etc. are just a few of the steps we can take to improve our system. If we don’t address these underlying issues, all the other data-driven techniques around slices and boosting will have propagated these issues and retraining won’t be as impactful as we may have intended.

## Impact

These are just a few of the dynamic levers that need to be evaluated and adjusted after nearly every iteration. So the brunt of the work comes after shipping the first model and when trying to keep our system relevant to the changing world (developing user preferences, social trends, pandemics, etc.). So, all this work better be worth it, such that even incremental improvements translate to meaningful impact! But the good news is that we’ve created a highly transparent system that can adapt to any change with fewer and fewer technical interventions, thanks to both the increased data to learn from and the learned evaluation, monitoring and update policy criteria in place.

!!! warning
    While it's important to iterate and optimize the internals of our workflows, it's even more important to ensure that our ML systems are actually making an impact. We need to constantly engage with stakeholders (management, users) to iterate on **why** our ML system exists.

## Tooling

Tooling is improving ([monitoring](monitoring.md){:target="_blank"}, [evaluation](testing.md){:target="_blank"}, [feature store](feature-store.md){:target="_blank"}, etc.) & control planes (ex. [Metaflow](https://metaflow.org/){:target="_blank"}) are emerging to connect them. Even tooling to allow domain experts to interact with the ML system beyond just labeling (dataset curation, segmented monitoring investigations, explainability). A lot of companies are also building centralized ML platforms (ex. Uber's [Michelangelo](https://eng.uber.com/michelangelo-machine-learning-platform/){:target="_blank"} or LinkedIn's [ProML](https://engineering.linkedin.com/blog/2019/01/scaling-machine-learning-productivity-at-linkedin){:target="_blank"}) to allow their developers to move faster and not create redundant workflows (shared feature stores, health assurance monitoring pipelines, etc.)

<!-- Citation -->
{% include "cite.md" %}