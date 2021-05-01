---
template: lesson.html
title: Designing Solutions for ML Systems
description: Designing a solution with constraints.
keywords: solutions, design, systems, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/MLOps
video: https://www.youtube.com/watch?v=Gi1VlFV8e_k
---

<!-- <div class="ai-center-all">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/Gi1VlFV8e_k?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div> -->

{% include "styles/lesson.md" %}

## Intuition

Once we've identified our main objective, we can hypothesize solutions using a three-step process: visualize, understand and design.

## Visualize
Visualize an ideal solution to our problem **without** factoring in constraints. It may seem like a waste of time to think freely without constraints, but it's a chance to think creatively.

- most creative solutions start from a blank slate
- void of the bias from previous approaches

## Understand
Understand how the problem is currently being solved (if at all) and the **how** and **why** things are currently done the way they are.

- prevents us from reinventing the wheel
- gives insight into processes and signals
- opportunity to question everything

## Design
Design from our ideal solution while factoring in **constraints**.

### Automate or augment?

- be wary of completely removing the user
- transition from augment to automate as trust grows

### UX constraints

- privacy, personalization, property
- dictate the components of our solution

### Technical constraints

- data, time, performance, cost, interpretability, latency
- dictate the complexity of our solutions

!!! note
    The main goal here is to think like a problem solver, as opposed to a naive model fitter.

<center>

| ❌&nbsp;&nbsp; Model fitter    | ✅&nbsp;&nbsp; Problem solver                          |
| :---------- | :----------------------------------- |
| *naively maps* a set of inputs to outputs         | knows which set of inputs and outputs are *worth mapping*             |
| obsesses on *methods* (models, SOTA, single metric, etc.)  | focuses on *product* (objective, constraints, evaluation, etc.)   |
| fitting methods are *ephemeral*        | foundational mental models are *enduring*        |

</center>

## Evaluation

Before we start building our solution, we need to make sure we have methods to evaluate it. We'll use our objective here to determine the evaluation criteria.

- be clear about what metrics you are prioritizing
- be careful not to over optimize on any one metric

!!! note
    We should also apply our metrics across various [slices](evaluation.md#slices){:target="_blank"} of data (timestamps, classes, features, etc.) because the overall performance can be very different from granular performance. This is especially important if certain slices of data are more important or if daily performance is more meaningful than overall (rolling) performance. We'll take a closer look at this in our [testing](testing.md){:target="_blank"} and [monitoring](monitoring.md){:target="_blank"} lessons.


Evaluation doesn't just involve measuring how well we're doing but we also need to think about what happens when our solution is incorrect.

- what are the fallbacks?
- what feedback are we collecting?

## Application

Our main objective is to allow users to discover the precise resource.

1. **Visualize** The ideal solution would be to ensure that all projects have the proper metadata (tags) so users can discover them.

2. **Understand** So far users search for projects using tags. It's important to note here that there are other available signals about each project such as the title, description, details, etc. which are not used in the search process. So this is good time to ask why we only rely on tags as opposed to the full text available? Tags are added by the project's author and they represent core concepts that the project covers. This is more meaningful than keywords found in the project's details because the presence of a keyword does not necessarily signify that it's a core concept. Additionally, many tags are inferred and don't explicitly exist in the metadata such as `natural-language-processing`, etc. But what we will do is use the other text metadata to determine relevant tags.

3. **Design** So we would like all projects to have the appropriate tags and we have the necessary information (title, description, etc.) to meet that requirement.

    - `#!js Augment vs. automate`: We will decide to augment the user with our solution as opposed to automating the process of adding tags. This is so we can ensure that our suggested tags are in fact relevant to the project and this gives us an opportunity to use the author's decision as feedback for our solution.
    - `#!js UX constraints`: We also want to keep an eye on the number of tags we suggest because suggesting too many will clutter the screen and overwhelm the user.
    - `#!js Tech constraints`: we will need to maintain low latency (>100ms) when providing our generated tags since authors complete the entire process within a minute.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/solution/suggested_tags.png" width="550" alt="pivot">
</div>
<div class="ai-center-all">
  <small>UX of our hypothetical solution</small>
</div>

!!! note
    For the purpose of this course, we're going to develop a solution that involves applied machine learning in production. However, we would also do A/B testing with other approaches such as simply altering the process where users add tags to projects. Currently, the tagging process involves adding tags into an input box but what if we could separate the process into sections like `frameworks`, `tasks`, `algorithms`, etc. to guide the user to add relevant tags. This is a simple solution that needs to be tested against other approaches for effectiveness.

As for evaluating our solution, we want to be able to suggest highly relevant tags (precision) so we don't fatigue the user with noise. But *recall* that the whole point of this task is to suggest tags that the author will miss (recall) so we can allow our users to find the best resource! So we'll need to tradeoff between precision and recall.

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

Normally, the goto option would be the F1 score (weighted precision and recall) but we shouldn't be afraid to craft our own evaluation metrics that best represents our needs. For example, we may want to account for both precision and recall but give more weight to recall. We may also want to evaluate performance at various levels such as for specific classes or [slices](evaluation.md#slices){:target="_blank"} of data.

We may also want to consider separating our test set before shuffling and evaluating daily (or any window of time) metrics as opposed to an overall (rolling) basis. This might give us insight into how our model may actually perform on a daily basis once deployed and catch degradations earlier. We'll cover these concepts in our [monitoring](monitoring.md){:target="_blank"} lesson.

Fortunately in our application, when we make a mistake, it's not catastrophic. The author will simply ignore it but we'll capture the error based on the tags that the author does add. We'll use this feedback (in addition to an annotation workflow) to improve on our solution over time.

!!! note
    If we want to be very deliberate, we can provide the authors an option to report erroneous tags. Not everyone may act on this but it could reveal underlying issues we may not be aware of.


## Resources
- [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/research/uploads/prod/2019/01/AI-Guidelines-poster_nogradient_final.pdf){:target="_blank"}
- [People + AI Guidebook](https://pair.withgoogle.com/guidebook/){:target="_blank"}


<!-- Citation -->
{% include "cite.md" %}