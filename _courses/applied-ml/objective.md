---
layout: page
title: Objective · Applied ML
description: Defining the core objective of our task.
image: /static/images/applied_ml.png

course-url: /courses/applied-ml/
next-lesson-url: /courses/applied-ml/solution/
---

<!-- Header -->
<div class="row">
  <div class="col-md-8 col-6 mr-auto">
    <h1 class="page-title">{{ page.title | split: " · " | first }}</h1>
  </div>
  <div class="col-md-4 col-6">
    <div class="btn-group float-right mb-0" role="group">
      <a href="{% link index.md %}" class="btn btn-sm btn-outline-secondary"><i
          class="fas fa-sm fa-arrow-left mr-1"></i>Return home</a>
    </div>
  </div>
</div>
<hr class="mt-0">

<!-- Video -->
<div class="ai-center-all mt-2">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/_sYrVHGRqPo?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div>
<div class="ai-center-all mt-2">
  <small>Accompanying video for this lesson. <a href="https://www.youtube.com/madewithml?sub_confirmation=1" target="_blank">Subscribe</a> for updates!</small>
</div>

<h3><u>Intuition</u></h3>
Identifying the objective should always be the first step when solving any problem. It acts as the **guide** for all subsequent decision making and will prevent us from getting distracted along the way. However, identifying the objective isn't always straightforward, especially when we aren't analyzing the problem through the appropriate lens.

> A proven way to identify the key objective is to think about the problem from the **user's perspective** so that we're positioned to think about the underlying issue as opposed to technological shortcomings.

<!-- <i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [0:00](https://www.youtube.com/watch?v=_sYrVHGRqPo&list=PLqy_sIcckLC2jrxQhyqWDhL_9Uwxz8UFq&index=1&t=0s){:target="_blank"} for a video walkthrough of this section. -->

<h3><u>Application</u></h3>
In our application, we have a set of projects (with tags) that users search for (using tags).

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

> This is analogous to development in ML where you can iterate on model architectures (to gain incremental improvements) but we can gain massive improvements by improving the quality of your underlying dataset.

<h3><u>Resources</u></h3>
- [Know Your Customers’ “Jobs to Be Done”](https://hbr.org/2016/09/know-your-customers-jobs-to-be-done){:target="_blank"}

<!-- Footer -->
<hr>
<div class="row mb-4">
  <div class="col-6 float-left">
    <a href="{% link index.md %}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-left mr-1"></i>Return home</a>
  </div>
  <div class="col-6">
    <div class="float-right">
      <a href="{{ page.next-lesson-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-right mr-1"></i>Next lesson</a>
    </div>
  </div>
</div>

