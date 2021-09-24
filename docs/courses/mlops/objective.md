---
template: lesson.html
title: Outlining Objectives for ML Systems
description: Defining the core objective of our task.
keywords: objectives, product management, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
video: https://www.youtube.com/watch?v=_sYrVHGRqPo
---

<!-- <div class="ai-center-all">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/_sYrVHGRqPo?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div> -->

{% include "styles/lesson.md" %}

## Intuition

Identifying the objective should always be the first step when solving any problem. It acts as the **guide** for all subsequent decision making and will prevent us from getting distracted along the way. However, identifying the objective isn't always straightforward, especially when we aren't analyzing the problem through the appropriate lens.

!!! question "When you're given a problem to solve..."

    1. How is the problem presented to you (vague/specific, business/technical)?

    2. What type of problems do you prefer?

    3. What can you change if the problem isn't presented this way?

        ??? quote "Show answer"
            - Ask appropriate questions to reframe the problem.
            - Ask to be involved in the problem formulation stage earlier on for more context.

    4. How can you identify the actual objective from the problem?

        ??? quote "Show answer"
            - break down the problem
            - understand the problem from the user's perspective

    5. What are other aspects of the problem to consider?

        ??? quote "Show answer"
            - Resources (time, effort, etc.)
            - Relevant data (signal) to work on objective?
            - Can you push back on the problem to solve?
            - Is your derived objective aligned with the business?
            - Is your derived objective ethical?


## Application
In our application, we have a set of [projects](https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/projects.json){:target="_blank"} (with tags) that users search for (using tags).

```json linenums="1"
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

Our assigned problem is to:

> **Improve search and discoverability** because users have complained that they are not able to discover the appropriate resources..

Based on key words in the problems, it's easy to prematurely jump to technological objectives such as:

- we need a better search algorithm
- we need better search infrastructure
- we need a sleeker search interface

Though some of these objectives may be valid, they may not resolve the underlying issue. What we need to think about is **why** the *user* isn't able to discover the right resource.

- What exactly are the user complaints?
- Is it an issue of content presence or discoverability?
- Any specifics on how exactly management wants to improve search?
- What past data do we have to work with? Are the issues flagged?

Once we have a clear objective defined, we can start to think about potential [solutions](solution.md){:target="_blank"}.

## Resources
- [Know Your Customers’ “Jobs to Be Done”](https://hbr.org/2016/09/know-your-customers-jobs-to-be-done){:target="_blank"} by [Clayton M. Christensen](https://en.wikipedia.org/wiki/Clayton_Christensen){:target="_blank"}


<!-- Citation -->
{% include "cite.md" %}