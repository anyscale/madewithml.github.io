---
description: Defining the core objective of our task.
image: https://madewithml.com/static/images/applied_ml.png
---

:octicons-mark-github-16: [Repository](https://github.com/GokuMohandas/applied-ml){:target="_blank"} · :octicons-device-camera-video-24: [Video](https://www.youtube.com/watch?v=_sYrVHGRqPo){:target="_blank"}

Defining the core objective of our task.

<div class="ai-center-all">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/_sYrVHGRqPo?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div>

## Intuition
Identifying the objective should always be the first step when solving any problem. It acts as the **guide** for all subsequent decision making and will prevent us from getting distracted along the way. However, identifying the objective isn't always straightforward, especially when we aren't analyzing the problem through the appropriate lens.

!!! note
    A proven way to identify the key objective is to think about the problem from the **user's perspective** so that we're positioned to think about the underlying issue as opposed to technological shortcomings.

## Application
In our application, we have a set of [projects](https://raw.githubusercontent.com/GokuMohandas/applied-ml/main/datasets/projects.json){:target="_blank"} (with tags) that users search for (using tags).

```json
{
    "id": 2427,
    "title": "Knowledge Transfer in Self Supervised Learning",
    "description": "A general framework to transfer knowledge from deep self-supervised models to shallow task-specific models.",
    "tags": [
        "article",
        "tutorial",
        "knowledge-distillation",
        "model-compression",
        "self-supervised-learning"
    ]
}
```

Our assigned task is to *improve search and discoverability*. We shouldn't prematurely jump to technological objectives such as:

- we need a better search algorithm
- we need better search infrastructure
- we need a sleeker search interface

Though some of these objectives may be valid, they may not resolve the underlying issue. What we need to think about is why the *user* isn't able to discover the right resource. This becomes our core objective and we'll further refine it in the next lesson when we design our solution.

!!! note
    This is analogous to development in ML where you can iterate on model architectures (to gain incremental improvements) but we can gain massive improvements by improving the quality of your underlying dataset.

## Resources
- [Know Your Customers’ “Jobs to Be Done”](https://hbr.org/2016/09/know-your-customers-jobs-to-be-done){:target="_blank"}

