---
layout: page
title: Evaluation · Applied ML in Production
description: Determining how well our solution is performing over time.
image: /static/images/courses/applied-ml-in-production/evaluation.png
tags: product product-management evaluation

course-url: /courses/applied-ml-in-production/
next-lesson-url: /courses/applied-ml-in-production/iteration/
---

<!-- Header -->
<div class="row">
  <div class="col-md-8 col-6 mr-auto">
    <h1 class="page-title">{{ page.title | split: " · " | first }}</h1>
  </div>
  <div class="col-md-4 col-6">
    <div class="btn-group float-right mb-0" role="group">
      <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i
          class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
    </div>
  </div>
</div>
<hr class="mt-0">

<h3><u>Intuition</u></h3>
Before we start building our solution, we need to make sure we have methods to evaluate it. We'll use our objective here to determine the evaluation criteria.
- be clear about what metrics you are prioritizing
- be careful not to over optimize on any one metric

Evaluation doesn't just involve measuring how well we're doing but we also need to think about what happens when our solution is incorrect.
- what are the fallbacks?
- what feedback are we collecting?

<h3><u>Application</u></h3>
For our task, we want to be able to suggest highly relevant tags (precision) so we don't fatigue the user with noise. But *recall* that the whole point of this task is to suggest tags that the author will miss (recall) so we can allow our users to find the best resource! So we'll need to tradeoff between precision and recall.

<figure>
  <img src="/static/images/courses/applied-ml-in-production/confusion_matrix.png" width="550" alt="pivot">
  <figcaption>Confusion matrix [<a href="https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks" target="_blank">source</a>]</figcaption>
</figure>

Normally, the goto option would be the F1 score (weighted precision and recall) but we shouldn't be afraid to craft our own evaluation metrics that best represents our needs. For example, we may want to account for both precision and recall but give more weight to recall.

<figure>
  <img src="/static/images/courses/applied-ml-in-production/metrics_table.png" width="550" alt="pivot">
  <figcaption>Sample of different evaluation metrics [<a href="https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks" target="_blank">source</a>]</figcaption>
</figure>

Fortunately, when we make a mistake, it's not catastrophic. The author will simply ignore it but we'll capture the error based on the tags that the author does add. We'll use this feedback (in addition to an annotation workflow) to improve on our solution over time.

> If we want to be very deliberate, we can provide the authors an option to report erroneous tags. Not everyone may act on this but it could reveal underlying issues we may not be aware of.

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