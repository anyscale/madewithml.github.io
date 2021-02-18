---
template: lesson.html
title: Evaluating our ML Systems
description: Determining how well our solution is performing over time.
keywords: evaluation, applied ml, mlops, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
---

:octicons-mark-github-16: [Repository](https://github.com/GokuMohandas/applied-ml){:target="_blank"} · :octicons-device-camera-video-24: [Video](https://www.youtube.com/watch?v=AwajdDVR_C4){:target="_blank"}

<!-- <div class="ai-center-all mt-2">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/AwajdDVR_C4?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div> -->

## Intuition

Before we start building our solution, we need to make sure we have methods to evaluate it. We'll use our objective here to determine the evaluation criteria.

- be clear about what metrics you are prioritizing
- be careful not to over optimize on any one metric

!!! note
    We should also apply our metrics across various slices of data (timestamps, classes, features, etc.) because the overall performance can be very different from granular performance. This is especially important if certain slices of data are more important or if daily performance is more meaningful than overall (rolling) performance. We'll take a closer look at this in our [testing](testing.md){:target="_blank"} and [monitoring](monitoring.md){:target="_blank"} lessons.


Evaluation doesn't just involve measuring how well we're doing but we also need to think about what happens when our solution is incorrect.

- what are the fallbacks?
- what feedback are we collecting?

## Application
For our task, we want to be able to suggest highly relevant tags (precision) so we don't fatigue the user with noise. But *recall* that the whole point of this task is to suggest tags that the author will miss (recall) so we can allow our users to find the best resource! So we'll need to tradeoff between precision and recall.

$$ \text{accuracy} = \frac{TP+TN}{TP+TN+FP+FN} $$

$$ \text{recall} = \frac{TP}{TP+FN} $$

$$ \text{precision} = \frac{TP}{TP+FP} $$

$$ F_1 = 2 * \frac{\text{precision }  *  \text{ recall}}{\text{precision } + \text{ recall}} $$

<center>

| Variable    | Description                          |
| :---------- | :----------------------------------- |
| $TP$         | # of samples truly predicted to be positive and were positive         |
| $TN$         | # of samples truly predicted to be negative and were negative         |
| $FP$         | # of samples falsely predicted to be positive but were negative       |
| $FN$         | # of samples falsely predicted to be negative but were positive       |

</center>

Normally, the goto option would be the F1 score (weighted precision and recall) but we shouldn't be afraid to craft our own evaluation metrics that best represents our needs. For example, we may want to account for both precision and recall but give more weight to recall. We also may want to consider separating our test set before shuffling and evaluating daily (or any window of time) metrics as opposed to an overall (rolling) basis. This might give us insight into how our model may actually perform on a daily basis once deployed and catch degradations earlier. We'll cover these concepts in our [monitoring](monitoring.md){:target="_blank"} lesson.

Fortunately, when we make a mistake, it's not catastrophic. The author will simply ignore it but we'll capture the error based on the tags that the author does add. We'll use this feedback (in addition to an annotation workflow) to improve on our solution over time.

!!! note
    If we want to be very deliberate, we can provide the authors an option to report erroneous tags. Not everyone may act on this but it could reveal underlying issues we may not be aware of.


<!-- Citation -->
{% include "cite.md" %}