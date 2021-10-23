---
template: lesson.html
title: Project Management for Machine Learning
description: A template to guide the development cycle for machine learning systems that factors in product requirements, design and project considerations that are integral to planning and developing.
keywords: project management, product management, design docs, scoping, management, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
---

{% include "styles/lesson.md" %}

## Template

While this template should be completed in sequential order but will naturally involve nonlinear engagement based on feedback:

[Progress](#progress) (*Where*) → [Product](#product-management) (*What* & *Why*) → [Methodology](#methodology) (*How*) → [Project](#project-management) (*Who* & *When*)

> This is not an exact template to follow but is a guide for important aspect to consider when planning and designing your ML projects.

And each of these sections should follow these best practices and should be overseen by the directly responsible individual (**DRI**):

- **Details**: each section should be thoroughly written using insights from relevant team members, research, etc. This often involves white boarding sessions to break down the intricate details and then writing about them to easily convey them to others.
- **Feedback**: each section should also be approved by relevant stakeholders. This can be done iteratively by performing a canary feedback rollout to ensure there are no open unanswered questions before engaging with executives.
- **Updates**: this documentation should always be kept up-to-date so new (and current) members can always refer to it in the event of on-boarding, conflict, validation, etc.

!!! warning
    We shouldn’t try to create one master document for all versions of our product. Each version should have its own documentation with what’s in and out of scope so we can focus all the discussion around what’s immediately relevant.

## Progress

[*Where*]: status of where we are in the project in terms of timeline? This is to just provide a quick update on progress and relevant links for those new to it.

### Current state

> What is the status of the team’s progress and what release are they working on which should be linked to the appropriate version details below.

On-schedule | v1.5

### Resources

> Include all relevant links to key presentations, supplementary documentation, contact info, etc.

[Exec Keynote]() | [PRD](https://en.wikipedia.org/wiki/Product_requirements_document){:target="_blank"} | [Project plan](https://en.wikipedia.org/wiki/Project_plan){:target="_blank"} | [RFC](https://en.wikipedia.org/wiki/Request_for_Comments){:target="_blank"} | [DRI Contact](mailto:hello@madewithml.com){:target="_blank"}

## Product management

[*What* & *Why*]: motivate the need for the product and outline the objectives and key results.

### Overview

> Describe the feature/problem at a high level. This is the section where you’re setting the scene for someone new to the topic so avoid getting into the details until you reach the sections further below.

We are a service that has content (with tags) with a rudimentary search mechanism that involves searching by tags. Recently, there has been an increase in complaints pertaining to the search and discovery experience. Upon further investigation, the vast majority of the complaints are a result of content having insufficient or irrelevant tags.

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

### Relevance

> Why is this feature/problem important and why now? Talk about experiences leading to its discovery and the impact on the business. Out of all the other problems we could be addressing right now, why is this problem the one that’s worth solving right now? Justify using relative impact on business compared to other issues in the backlog.

**Core business values**
One of our core business values is to provide the curation that our users are not able to find anywhere else. Therefore, it’s a top priority to ensure that we meet this core value.

**Engagement**
When our users are able to discover the precise resources for their needs, this drives engagement on our platform and improves perceived value. If we had addressed the search related complaints over the last X months, it would have increased engagement by X% leading to Y% increase in sponsorship fees.


### Background
Describe any relevant background information relevant to this project, specially aspects that may not be so intuitional.

- **Q**: Why are tags even used?

    - Tags are added by the project's author and they represent core concepts that the project covers.
    - Keywords in a project's description does not necessarily signify that it's a core concept.

- **Q**: Why can tags provide that keywords in full-text cannot?

    - Many tags are inferred and don't explicitly exist in the metadata, such as `natural-language-processing`, and users will definitely use these in their search terms for broad search and filtering.

We have also validated that the vast majority of user complaints step from missing implicit tags and so a full-text-based approach would not address this underlying issue.

!!! question "What to focus on first?"
    What can you do if your solution architecture process unravels system issues in your team. Do you ignore those and do your best or punt the problem until they're addressed?

    ??? quote "Show answer"
        It depends. By understanding the constraints, you'll unravel systemic issues in your team. Such as legacy systems introducing bugs or simple systems that have yet to be build (ex. full-text search). At this point, you need to have a discussion with your team on whether the system change needs to worked on first because it might remove the problem all together (can cause tension/conflict!)

### Solutions

> Describe current solutions, feasible solutions and alternative approaches considered. Start with the most obvious solutions and ask questions to decouple the objectives.

- Current solutions → None
- Predict tags for a given content → use available signals from content.
- Measure engagement on platform → track engagement after discovery of content.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/solution/suggested_tags.png" width="550" alt="suggested tags">
</div>
<div class="ai-center-all">
  <small>UX of our hypothetical solution</small>
</div>

### Feasibility

> Our solutions have so many nuances tied to them that will be revealed when we break down the proposals. This will also help our project team scope these into separate releases that will each provide end-to-end value. We should also note potential data dependencies **[DATA]** and explorations **[EXPLORE]** required to assess the feasibility of our proposals.

What are the signals?

- title, description and other relevant metadata from the content **[DATA]**
- are the tokens in the content metadata enough signal to predict implicit tags **[EXPLORE]**

Alternatives considered

- Currently, the tagging process involves adding tags into an input box but what if we could separate the process into sections like `frameworks`, `tasks`, `algorithms`, etc. to guide the user to add relevant tags.

!!! warning
    We would definitely implement this simple alternative first and continue to use it for A/B testing to benchmark against more sophisticated methods. But for this course, we're going to jump the ML-based solution so we can design and implement components that are particular to ML systems.

### Constraints

Discuss the constraints that we have to account for in our solutions. A large majority of constraints can directly come from our service-level agreements (SLAs) with customers and internal systems regarding time, $, performance, latency, infrastructure, privacy, security, UI/UX.

- maintain low latency (>100ms) when providing our generated tags. **[Latency]**
- avoid overwhelming the author with too many predicted tags. **[UI/UX]**

!!! question "Converting constraints to timelines"
    How can we best estimate the time it will take to develop our solutions while accounting for all these constraints? And for the sake of this course, we'll assume that our solution involves machine learning. How can we predict how long an experimental and iterative process can take?

    ??? quote "Show answer"

        - create a plan outlining the problem proposed solutions and your plan for each part of the design process (each part = units in this MLOps course, ie. testing, monitoring, etc.)
        - educate and set expectations with leadership because ML solutions involve a lot of trial and error.
        - have frequent meetings at the start of a project so that your team can quickly pivot (very important in this non-linear path with data-driven solutions).
        - architect initial solution to be deterministic and close the loop (more in [iteration lesson](iteration.md){:target="_blank"})
        - subsequent projects can leverage well tested code and systems, so be sure to account for this.
        - your judgement improves as you develop solution in this problem space and understand its nuances ([testing](testing.md){:target="_blank"}, [monitoring](monitoring.md){:target="_blank"}, etc.)

### Integration

How does this effort integrate with the current system and what additional work is needed for it? Our project team will use this to request for comments from appropriate team members and allocate resources in the releases. These integrations may also require separate documentation around specification for involved teams that can include wireframes, user stories, mock-ups, etc.

- **dependencies**: cluster resources to maintain and scale microservice based on demand.
- **consumers**: content creation/update UI to consume and display predicted tags. *[MOCK]*

### Requirements

> Describe core requirements, what will be articulated to the public, what user’s engagement would look like (users stories), etc. These details will help shape the functionality that all releases will have and be used as a guide for trade-offs (metrics, costs, etc.)

<table>
<thead>
  <tr>
    <th>Requirement</th>
    <th>Priority</th>
    <th>Release</th>
    <th>Status</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Identify explicit tags using rule-based text matching.</td>
    <td>High</td>
    <td>v1</td>
    <td>Complete</td>
  </tr>
  <tr>
    <td>Predict explicit and implicit tags for given content metadata.</td>
    <td>High</td>
    <td>v1</td>
    <td>Complete</td>
  </tr>
  <tr>
    <td>UI to poll prediction service when relevant content metadata is changed during content creation/update process.</td>
    <td>High</td>
    <td>v1</td>
    <td>Complete</td>
  </tr>
  <tr>
    <td>Allow authors to click on the recommended tag to automatically add it to their content.</td>
    <td>Medium</td>
    <td>v1</td>
    <td>In-progress</td>
  </tr>
  <tr>
    <td>Remove predicted tags if the author has already manually included them.</td>
    <td>Medium</td>
    <td>v1</td>
    <td>In-progress</td>
  </tr>
  <tr>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr>
</tbody>
</table>

!!! warning
    Ideally we wouldn't be doing both rule-based text matching and predicting implicit tags in the same release because they’re both different and involved requirements. In fact, we may use the identification release to create a properly labeled dataset to use for prediction. But, we’ll include it in this release so we can talk about some aspects of project management that’s unique to ML systems.

### Out of scope

> What aspects of the feature/problem should we not be concerned with for the immediate planning?

- using text from content metadata besides title and description, such as full-text HTML from associated links.
- interpretability for why we recommend certain tags.


### Decisions

> Given the feasibility evaluation, constraints and integration, what are the key decisions that need to be made? A recommended framework to use is driver, approved, contributor and informed (DACI) responsibility assignment matrix.

<table>
<thead>
  <tr>
    <th>Decision</th>
    <th>Approver</th>
    <th>Contributor</th>
    <th>Informed</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Using only the title and description as relevant signal for recommending tags</td>
    <td>Product DRI</td>
    <td>Data scientist</td>
    <td>Executives, external stakeholders, etc.</td>
  </tr>
  <tr>
    <td>Recommend tags from our approved list of tags only</td>
    <td>Product DRI</td>
    <td>Product DRI</td>
    <td>Executives, external stakeholders, etc.</td>
  </tr>
  <tr>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr>
</tbody>
</table>

## Methodology

[*How*]: describe our sequential approach towards building the product.

### Data

> What sources of relevant data do we need and what are the workflows we need to establish around data retrieval and [validation](testing.md#data){:target="_blank"}?

- **Sources**: access to a labeled and validated dataset with content metadata and all relevant tags.
- **Workflows**: access to live content metadata as an author is creating content.

### Evaluation

> Regardless of our version of implementation, we need to be able to evaluate how we’re performing. This evaluation criteria needs to be comprehensive and mimic what we can expect in production as much as possible. If we do update our evaluation criteria, we should apply it to all previous approaches so that we can objectively compare them. And if we have multiple metrics to evaluate on, we should **prioritize** what is most important. In many situations, this is also the section where we would discuss [offline](evaluation.md){:target="_blank"} and [online](monitoring.md#performance){:target="_blank"} evaluation methods.

#### Metrics
We want to be able to suggest highly relevant tags (precision) so we don't fatigue the user with noise. But *recall* that the whole point of this task is to suggest tags that the author will miss (recall) so we can allow our users to find the best resource! So we'll need to tradeoff between precision and recall. Normally, a good goto option would be the F1 score (weighted precision and recall) but we shouldn't be afraid to craft our own evaluation metrics that best represents our needs. For example, we may want to account for both precision and recall but give more weight to recall. We may also want to evaluate performance for specific classes or [slices](evaluation.md#slices){:target="_blank"} of data.

!!! question "What are your priorities"
    For your respective industries or areas of interest, do you know where the priorities are (metrics, errors and other tradeoffs)?

    ??? quote "Show answer"
        It entirely depends on the specific task. For example, in an email spam detector, precision is very important because it's better than we some spam then completely miss an important email. Overtime, we need to [iteration](iteration.md){:target="_blank"} on our solution so all evaluation metrics improve but it's important to know which one's we can't comprise on from the get-go.

#### Offline vs. online
For offline evaluation, we'll need a gold standard labeled dataset that we can use to benchmark all of our [methods](#methods), while for online evaluation we can initially use the signal on whether the author used our suggested tags and then we can adjust the labels after that new data goes through a proper QA pipeline.

#### Implications
Fortunately in our application, when we make a mistake, it's not catastrophic. The author will simply ignore it but we'll capture the error based on the tags that the author does add. We'll use this feedback (in addition to an annotation workflow) to improve on our solution over time.

### Methods

> Describe the different approaches that each deliver end-to-end utility while ensuring that they are well [tested](testing.md#models){:target="_blank"} and [evaluated](evaluation.md){:target="_blank"}.

!!! note
    Good principles to abide by here include:

    - **End-to-end utility**: the end result from every iteration should deliver minimum end-to-end utility so that we can benchmark iterations against each other and plug-and-play with the system.
    - **Keep it simple stupid (KISS)**: start from the simplest solution and slowly add complexity with justification along the way → [baselines](https://madewithml.com/courses/mlops/baselines){:target="_blank"}.
    - **Manual before ML**: incorporate deterministic components where we define the rules before using probabilistic ones that infer rules from data.
    - **Augment vs. automate**: allow the system to supplement the decision making process as opposed to making the final decision.
    - **Internal vs. external**: not all early releases have to be end-user facing. We can use early versions for internal validation, feedback, data collection, etc.
    - **Tested**: every approach needs to be well [tested](testing.md){:target="_blank"} (code, data + models) with the appropriate reporting so we can objectively benchmark different approaches.

For the purpose of this course, we're going to develop a solution that involves machine learning from the very beginning. However, we would also do [A/B testing](infrastructure.md#ab-tests){:target="_blank"} with other approaches such as simply altering the process where users add tags to projects. Currently, the tagging process involves adding tags into an input box but what if we could separate the process into sections like `frameworks`, `tasks`, `algorithms`, etc. to guide the user to add relevant tags. This is a simple solution that needs to be tested against other approaches for effectiveness. Then, we would try rule-based approaches such as simple text matching before trying to predict relevant tags from content metadata.

The main goal here is to think like a problem solver and motivate the need for additional complexity, as opposed to a naive model fitter:

<center>

| ❌&nbsp;&nbsp; Model fitter    | ✅&nbsp;&nbsp; Problem solver                          |
| :---------- | :----------------------------------- |
| *naively maps* a set of inputs to outputs         | knows which set of inputs and outputs are *worth mapping*             |
| obsesses on *methods* (models, SOTA, single metric, etc.)  | focuses on *product* (objective, constraints, evaluation, etc.)   |
| fitting methods are *ephemeral*        | reproducible systems are *enduring*        |

</center>

### Rollout

> What do the release strategies look like for our different versions? Note that not all releases have to be high stakes, external facing to the whole world. We can always include internal releases, gather feedback and iterate until we’re ready to increase the scope.

- Canary internal rollout, monitoring for proxy/actual performance, etc.
- Rollout to the larger internal team for more feedback.
- A/B rollout to a subset of the population to better understand UX, utility, etc.

### Feedback

> How do we receive feedback on our system and incorporate it into the next iteration? This can involve both human-in-the-loop feedback as well as automatic feedback via [monitoring](monitoring.md){:target="_blank"}, etc.

- use feedback from the internal rollout to figure out how to educate early users of the feature with new session hooks.
- use author's chosen tags as a proxy signal to quantify online performance.
- enforce human-in-loop checks on where they are conflicts between recommended tags and manually chosen tags.
- allow users to report issues related to suggested tags.

## Project management

[*Who* & *When*]: organizing all the product requirements into manageable timelines so we can deliver on the vision.

### Team

> Which teams and specific members from those teams need to be involved in this project? It’s important to consider even the minor features so that everyone is aware of it and so we can properly scope and prioritize our timelines. Keep in mind that this isn’t the only project that people might be working on.

- **Product**: the members responsible for outlining the product requirements and approving them may involve product managers, executives, external stakeholders, etc.
- **Methodology**:
    - Data engineering: these developers are often responsible for the data dependencies, which include robust workflows to continually deliver the data and ensuring that it’s properly validated and ready for downstream applications
    - Machine learning: develop the probabilistic systems with appropriate evaluation.
    - DevOps: deploy the application and help autoscale based on traffic.
    - UI/UX: consume the system’s outputs to deliver the new experience to the user.
    - Accessibility: help educate the community for the new rollouts and to assist with decisions around sensitive issues.
    - Site reliability: maintain the application and to potentially oversee that online evaluation/monitoring workflows are working as they should.
- **Project**: the members responsible for iterative engagement with the product and engineering teams to ensure that the right product is being built and that it’s being built appropriately may include project managers, engineering managers, etc.

### Timeline

> This is where the project scoping begins to take place. Often, the stakeholders will have a desired time to release and the functionality to be delivered. There  *will* be a lot of batch and forth on this based on the results from the feasibility studies, so it's very important to be thorough and transparent to set expectations quickly.

**v1**: predict relevant tags using content title and description metadata

- Exploration studies conducted by XX
- Pushed to dev for A/B testing by XX
- Pushed to staging with on-boarding hooks by XX
- Pushed to prod by XX


<small><sup>*</sup> extremely simplified timeline. An actual timeline would depict timelines from all the different teams stacked on top of each other with vertical lines at specified time-constraints or version releases.</small>

### Deliverables

We need to break down all the [requirements](#requirements) for a particular release into clear deliverables that specify the deliverable, contributors, dependencies, acceptance criteria and status. This will become the granular checklist that our teams will use to decide what to work on next and to ensure that they’re working on it properly (with all considerations).

<table>
<thead>
  <tr>
    <th>Requirement</th>
    <th>Priority</th>
    <th>Release</th>
    <th>Status</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Predict explicit and implicit tags for given content metadata.</td>
    <td>Medium</td>
    <td>v1</td>
    <td>Complete</td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th>Deliverable</th>
    <th>Contributors</th>
    <th>Dependencies</th>
    <th>Acceptance criteria</th>
    <th>Status</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Labeled dataset with content metadata and ground-truth tags</td>
    <td>Project DRI, data engineer</td>
    <td>Access to location of content with relevant metadata</td>
    <td>Validation of ground-truth labels</td>
    <td>TBD<br><br><br></td>
  </tr>
  <tr>
    <td>Trained model that can predict relevant tags given content metadata</td>
    <td>Data scientist</td>
    <td>Labeled dataset</td>
    <td>Versioned, reproducible, tested (coverage) and evaluation results.</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td>Scalable endpoint that can be used to retrieve relevant tags</td>
    <td>ML engineer, DevOps engineer</td>
    <td>Versioned, reproducible, tested and evaluated model</td>
    <td>Stress tests to ensure autoscaling capabilities.</td>
    <td>TBD</td>
  </tr>
  <tr>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr>
</tbody>
</table>

<!-- Citation -->
{% include "cite.md" %}