---
layout: page
title: Evaluation · Applied ML
description: Determining how well our solution is performing over time.
image: /static/images/applied_ml.png
redirect_from: /courses/applied-ml-in-production/evaluation/

course-url: /courses/applied-ml/
next-lesson-url: /courses/applied-ml/iteration/
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
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/AwajdDVR_C4?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div>
<div class="ai-center-all mt-2">
  <small>Accompanying video for this lesson. <a href="https://www.youtube.com/madewithml?sub_confirmation=1" target="_blank">Subscribe</a> for updates!</small>
</div>

<h3><u>Intuition</u></h3>
Before we start building our solution, we need to make sure we have methods to evaluate it. We'll use our objective here to determine the evaluation criteria.
- be clear about what metrics you are prioritizing
- be careful not to over optimize on any one metric

Evaluation doesn't just involve measuring how well we're doing but we also need to think about what happens when our solution is incorrect.
- what are the fallbacks?
- what feedback are we collecting?

<h3><u>Application</u></h3>
For our task, we want to be able to suggest highly relevant tags (precision) so we don't fatigue the user with noise. But *recall* that the whole point of this task is to suggest tags that the author will miss (recall) so we can allow our users to find the best resource! So we'll need to tradeoff between precision and recall.

$$ \text{accuracy} = \frac{TP+TN}{TP+TN+FP+FN} $$

$$ \text{recall} = \frac{TP}{TP+FN} $$

$$ \text{precision} = \frac{TP}{TP+FP} $$

$$ F_1 = 2 * \frac{\text{precision }  *  \text{ recall}}{\text{precision } + \text{ recall}} $$

* $$TP$$: # of samples truly predicted to be positive and were positive
* $$TN$$: # of samples truly predicted to be negative and were negative
* $$FP$$: # of samples falsely predicted to be positive but were negative
* $$FN$$: # of samples falsely predicted to be negative but were positive

Normally, the goto option would be the F1 score (weighted precision and recall) but we shouldn't be afraid to craft our own evaluation metrics that best represents our needs. For example, we may want to account for both precision and recall but give more weight to recall.

Fortunately, when we make a mistake, it's not catastrophic. The author will simply ignore it but we'll capture the error based on the tags that the author does add. We'll use this feedback (in addition to an annotation workflow) to improve on our solution over time.

> If we want to be very deliberate, we can provide the authors an option to report erroneous tags. Not everyone may act on this but it could reveal underlying issues we may not be aware of.

<!-- Footer -->
<hr>
<div class="row mb-4">
  <div class="col-6 mr-auto">
    <a href="{% link index.md %}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-left mr-1"></i>Return home</a>
  </div>
  <div class="col-6">
    <div class="float-right">
      <a href="{{ page.next-lesson-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-right mr-1"></i>Next lesson</a>
    </div>
  </div>
</div>