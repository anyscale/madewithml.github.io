---
template: lesson.html
title: Designing Solutions for ML Systems
description: Designing a solution with constraints.
keywords: solutions, design, systems, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
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

Once we've identified our main objective, we can hypothesize solutions using a three-step process: visualize, understand and design. Note that we're not talking about a technical solution just yet. In fact, we'll be using our three-step process to evaluate if we even need a technical solution.

## Visualize
Visualize an ideal solution to our problem **without** factoring in constraints. It may seem like a waste of time to think freely without constraints, but it's a chance to think creatively about solution from a clean slate.

## Understand
Understand how the problem is currently being solved (if at all) and the **how** and **why** things are currently done the way they are.

- prevents us from reinventing the wheel
- gives insight into processes and signals available
- opportunity to question everything

!!! question "Visualize before understand?"
    Does the order of these two steps matter?

    ??? quote "Show answer"

        We want to visualize an ideal solution before understanding current processes so we're not subjecting ourselves to any bias. Some of the most simplistic and creative solutions arise this way. *However*, this can be a frustrating step because you may already know that your constraint-free solution is not possible due to resources, legacy systems, etc. It's important to identify these and document them so you can advocate for the right next steps when the opportunity arises. Often, if your solution is successful, you may be asked to extend the improvements based on the business impact. Your constraint-free solution can come in handy in these situations.

## Design
Design from our ideal solution while factoring in constraints.

### Automate or augment?

Does our envisioned solution involve trying to completely remove the user from the problem (automate) or help the user make a better decision (augment)?

- be wary of completely removing the user
- transition from augment to automate as trust grows

### UX constraints

How will our solution be presented and what are aspects of the user experience that we need to respect?

- privacy, personalization, digital real estate
- dictate the components of our solution

### Technical constraints

What are the details around the team, data and system that you need to account for?

- availability of data, time, performance, cost, interpretability, latency
- dictate the complexity of our solutions

!!! question "Converting constraints to timelines"
    How can we best estimate the time it will take to develop our solutions while accounting for all these constraints? And for the sake of this course, we'll assume that our solution involves machine learning. How can we predict how long an experimental and iterative process can take?

    ??? quote "Show answer"

        - create a plan outlining the problem proposed solutions and your plan for each part of the design process (each part = units in this MLOps course, ie. testing, monitoring, etc.)
        - educate and set expectations with leadership because ML solutions involve a lot of trial and error.
        - have frequent meetings at the start of a project so that your team can quickly pivot (very important in this non-linear path with data-driven solutions).
        - architect initial solution to be deterministic and close the loop (more in [iteration lesson](iteration.md){:target="_blank"})
        - subsequent projects can leverage well tested code and systems, so be sure to account for this.
        - your judgement improves as you develop solution in this problem space and understand its nuances ([testing](testing.md){:target="_blank"}, [monitoring](monitoring.md){:target="_blank"}, etc.)

The main goal here is to think like a problem solver, as opposed to a naive model fitter.

<center>

| ❌&nbsp;&nbsp; Model fitter    | ✅&nbsp;&nbsp; Problem solver                          |
| :---------- | :----------------------------------- |
| *naively maps* a set of inputs to outputs         | knows which set of inputs and outputs are *worth mapping*             |
| obsesses on *methods* (models, SOTA, single metric, etc.)  | focuses on *product* (objective, constraints, evaluation, etc.)   |
| fitting methods are *ephemeral*        | reproducible systems are *enduring*        |

</center>

## Evaluation

Another key part of our solution is determining how to evaluate it and ensuring that it's reflective of our core objective.

- be clear about what metrics you are prioritizing
- be careful not to over optimize on any one metric

Evaluation doesn't just involve measuring how well we're doing but we also need to think about what happens when our solution is incorrect.

- what are the fallbacks?
- what feedback are we collecting?

> It's almost always a good idea to no directly pass the outputs of a machine learning system to the end user. Usually it involves converting probability distributions, filtering, etc. with some failure checks on top.

!!! question "What are your priorities"
    For your respective industries or areas of interest, do you know where the priorities are (metrics, errors and other tradeoffs)?

    ??? quote "Show answer"
        It entirely depends on the specific task. For example, in an email spam detector, precision is very important because it's better than we some spam then completely miss an important email. Overtime, we need to [iteration](iteration.md){:target="_blank"} on our solution so all evaluation metrics improve but it's important to know which one's we can't comprise on from the get-go.

## Application

Our main objective is to allow users to *discover* the precise resource. The main complaint was that users stumbled across projects that were relevant to their query but didn't show up as a search result. To address the issue around discoverability, we need to ensure that the projects have the proper tasks.

- **Visualize**
    - The ideal solution would be to ensure that all projects have the proper metadata (tags) so users can discover them. Ideally, I want this to be a mix of author input tags and generated tags based on context in case of human error.

- **Understand**

    - **Q**: Why are tags even used?
        - Tags are added by the project's author and they represent core concepts that the project covers.
        - Keywords in a project's description does not necessarily signify that it's a core concept.
    - **Q**: Why can tags provide that keywords in full-text cannot?

        ??? quote "Show answer"
            Many tags are inferred and don't explicitly exist in the metadata, such as `natural-language-processing`, and users will definitely use these in their search terms for broad search and filtering.

    - **Q**: Why can users only search using tags?
        - Full-text search is not implemented yet.

!!! question "Teams are messy..."
    What can you do if your solution architecture process unravels system issues in your team. Do you ignore those and do your best or punt the problem until they're addressed?

    ??? quote "Show answer"
        It depends. By understanding the constraints, you'll unravel systemic issues in your team. Such as legacy systems introducing bugs or simple systems that have yet to be build (ex. full-text search). At this point, you need to have a discussion with your team on whether the system change needs to worked on first because it might remove the problem all together (can cause tension/conflict!)

- **Design**

    - `#!js Augment vs. automate`: We will decide to augment the user with our solution as opposed to automating the process of adding tags. This is so we can ensure that our suggested tags are in fact relevant to the project and this gives us an opportunity to use the author's decision as feedback ([proxy signal](monitoring.md#performance){:target="_blank"}) for our solution.
    - `#!js UX constraints`: We also want to keep an eye on the number of tags we suggest because suggesting too many will clutter the screen and overwhelm the user.
    - `#!js Tech constraints`: we will need to maintain low latency (>100ms) when providing our generated tags since authors complete the entire process within a minute.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/solution/suggested_tags.png" width="550" alt="suggested tags">
</div>
<div class="ai-center-all">
  <small>UX of our hypothetical solution</small>
</div>

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

Normally, a good goto option would be the F1 score (weighted precision and recall) but we shouldn't be afraid to craft our own evaluation metrics that best represents our needs. For example, we may want to account for both precision and recall but give more weight to recall. We may also want to evaluate performance for specific classes or [slices](evaluation.md#slices){:target="_blank"} of data.

Fortunately in our application, when we make a mistake, it's not catastrophic. The author will simply ignore it but we'll capture the error based on the tags that the author does add. We'll use this feedback (in addition to an annotation workflow) to improve on our solution over time.

## Before ML

> Rule #1: Don’t be afraid to launch a product without machine learning. - [Google's Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml#rule_1_don%E2%80%99t_be_afraid_to_launch_a_product_without_machine_learning){:target="_blank"}

For the purpose of this course, we're going to develop a solution that involves machine learning from the very beginning. However, we would also do [A/B testing](infrastructure/#ab-tests){:target="_blank"} with other approaches such as simply altering the process where users add tags to projects. Currently, the tagging process involves adding tags into an input box but what if we could separate the process into sections like `frameworks`, `tasks`, `algorithms`, etc. to guide the user to add relevant tags. This is a simple solution that needs to be tested against other approaches for effectiveness.


## Resources
- [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/research/uploads/prod/2019/01/AI-Guidelines-poster_nogradient_final.pdf){:target="_blank"}
- [People + AI Guidebook](https://pair.withgoogle.com/guidebook/){:target="_blank"}
- [Google's Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}