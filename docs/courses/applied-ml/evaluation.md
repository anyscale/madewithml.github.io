---
description: Determining how well our solution is performing over time.
image: "/static/images/applied_ml.png"
---

:octicons-mark-github-16: [Repository](https://github.com/GokuMohandas/applied-ml){:target="_blank"} · :octicons-device-camera-video-24: [Video](https://www.youtube.com/watch?v=AwajdDVR_C4){:target="_blank"}

Determining how well our solution is performing over time.

<div class="ai-center-all mt-2">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/AwajdDVR_C4?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div>

## Intuition
Before we start building our solution, we need to make sure we have methods to evaluate it. We'll use our objective here to determine the evaluation criteria.
- be clear about what metrics you are prioritizing
- be careful not to over optimize on any one metric

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

Normally, the goto option would be the F1 score (weighted precision and recall) but we shouldn't be afraid to craft our own evaluation metrics that best represents our needs. For example, we may want to account for both precision and recall but give more weight to recall.

Fortunately, when we make a mistake, it's not catastrophic. The author will simply ignore it but we'll capture the error based on the tags that the author does add. We'll use this feedback (in addition to an annotation workflow) to improve on our solution over time.

!!! note
    If we want to be very deliberate, we can provide the authors an option to report erroneous tags. Not everyone may act on this but it could reveal underlying issues we may not be aware of.
