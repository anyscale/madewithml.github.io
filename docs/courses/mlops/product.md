---
template: lesson.html
title: Designing Machine Learning Products
description: A template to guide the development cycle for machine learning systems that factors in product requirements, design docs and project considerations.
keywords: project management, product management, design docs, scoping, management, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
---

{% include "styles/lesson.md" %}

## Overview

With so much content on machine learning these days, it's hard to keep things organized. We want to create a product that will automatically discover and classify content so that everything is organized for discovery. In this course, we'll not only develop the ML models but talk about all the important ML system and software design components required to put our model into production in a reproducible, reliable and robust manner.

We start this course by setting the scene for the precise product we'll be building. While this is a technical course, this initial product design process is everything. It's what creates great products that continue to improve over time. This lesson will offer the structure for how to think about ML + product. But before using the template, it's important to think about the following:

- **Company**: describe the company's core values, goals, user base, etc. This will be important to refer to when dealing with uncertainties.
- **Product**: what do we envision the product to do and why do we need it? We'll explore this in detail in the [product management](#product) section.
- **Releases**: how do we iteratively envision our product improve? Data, POCs, models, user experience, personalization, etc.
- **Concerns**: what are major concerns that our releases should be addressing? privacy, security, moderation, controversial decisions, etc.

## Template

This template is designed to guide machine learning product development. While this template will initially be completed in sequential order, it will naturally involve nonlinear engagement based on iterative feedback. We should follow this template for every major release of our products so that all the decision making is transparent and documented.

[Product](#product) (*What* & *Why*) → [System design](#system-design) (*How*) → [Project](#project) (*Who* & *When*)

Before we dive into the template, there are several details that apply to every section:

- **DRI**: each section should have directly responsible individuals (DRIs) appointed to own the components and to keep the respective documentation updated. These individuals are also important for communicating and highlighting key points during and across releases.
- **Details**: each section should be thoroughly written using insights from relevant team members, research, etc. This often involves white boarding sessions to break down the intricate details and then documenting them to share and implement.
- **Feedback**: each section should be reviewed and approved by relevant stakeholders. This can be done iteratively by performing a canary feedback rollout to ensure there are no open unanswered questions before engaging with executives.
- **Updates**: the documentation should always be kept up-to-date so new (and current) members can always refer to it in the event of on-boarding, conflict, validation, etc.

!!! note "Organizing design documentation"
    All great products improve over time and they require releases with clear objectives. In this lesson, we'll be designing the initial release but you would do the same for all future releases as well. The organizational structure of all this documentation would look like this:

    ```bash
    # Project scoping
    📂 project/
    ├── 📄 Overview
    ├── 📂 release-1
    | ├── 📄 product requirements [Product]
    | ├── 📄 design documentation [System design]
    | ├── 📄 project planning     [Project]
    ├── ...
    └── 📂 release-n
    ```

    It's important that we organize our projects this way so that there is one central location where all documentation can be found and updated. This also enforces that we scope releases appropriately and provide the thorough planning and documentation to execute each one.

## Product

[*What* & *Why*]: motivate the need for the product and outline the objectives and key results.

> Each section below has a dropdown component called "Our task", which will discuss the specific topic with respect to the specific product that we're trying to build.

### Overview

Describe the problem and product features at a high level. This is the section where you’re setting the scene for someone new to the topic, so avoid getting into the details until you reach the sections further below.

??? quote "Our task"

    Our objective is to create a system that can classify incoming ML content so that it's organized for discovery. To simplify our task, let's assume we already have a pipeline that delivers ML content from popular sources (Reddit, Twitter, etc.) and it's our job to classify these incoming streams.

    ```json linenums="1" title="Sample data point"
    {
        "id": 443,
        "created_on": "2020-04-10 17:51:39",
        "title": "AllenNLP Interpret",
        "description": "A Framework for Explaining Predictions of NLP Models",
        "tags": [
            "natural-language-processing"
        ],
        "text": "allennlp interpret framework explaining predictions nlp models"
    }
    ```

#### Relevance

Why is this important to work on and why now? We need to justify the efforts required with the potential impact on the business compared to other problems (backlog) we could be working on. As a general rule, it’s good to be as specific as possible in this section and use numerical values to strengthen claims.

??? quote "Our task"

    **Core business values**
    We want to be able to organize ML content and so being able to identify them correctly is crucial to our core business objective.

    **Engagement**
    When our users are able to discover the precise resources for their needs, this drives engagement on our platform and improves perceived value.


#### Background

Describe any background information relevant to this project, especially details that may not be so intuitive. This is also the section to mention previous approaches, competitive landscape, internal studies (with relevant summary of findings) and known obstacles.

??? quote "Our task"

    Our current search capabilities involve basic text matching with search queries. Unfortunately, many of the terms that our users are searching for involve queries that are not explicit in the content (ex. natural language processing). We need to be able to tag these implicit tags to all the content so that our users can discover them.

    - **Q**: Why do we even need tags?
        - Tags represent core concepts that may be implicit.
        - Keywords in a project's title or description may not be a core concept → noisy search results.

#### Objectives

What are the key objectives that we're trying to satisfy? These could be general objectives which we can then decouple or it can be specific success criteria/metrics.

??? quote "Our task"

    - Identify the relevant tags for a given content with 70% recall.
    - Reduce week-to-week tickets based on search errors by 20%.

!!! warning "Objectives vs. constraints"
    Objectives and [constraints](#constraints) are often incorrectly used interchangeably but they're very different. Objectives are things that we want to **achieve** that are under our control. Usually, they start with the phrase "We want &lt;objective&gt;" whereas constraints are **limitations** that we need to abide by and sound like "We can't do X because &lt;constraint&gt;". Another way to think about constraints is that it's something we wouldn't impose on ourselves if we didn't have to.

### Solutions

Describe current solutions and alternative approaches that our teams considered. These solutions may also require separate documentation around specification for involved teams that can include wireframes, user stories, mock-ups, etc.

??? quote "Our task"

    - Current solutions → simple text matching based search (noisy and incomplete).
    - Identify tags for a given content → use available signals from content.
    - Measure engagement on platform → track engagement after discovery of content.

    <div class="ai-center-all">
        <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/solution/suggested_tags.png" width="550" alt="suggested tags">
    </div>
    <div class="ai-center-all">
    <small>UX of our hypothetical solution</small>
    </div>

    - Alternatives considered
        - Currently, the tagging process involves adding tags into an input box but what if we could separate the process into sections like `frameworks`, `tasks`, `algorithms`, etc. to guide the user to add relevant tags. This is a very simple solution that should be A/B tested on and could even complement more sophisticated solutions.

#### Feasibility

Our solutions have so many nuances tied to them that will be revealed when we decouple the objectives. This will also help our project team scope these into separate releases that will each provide end-to-end value. We should also note potential data dependencies **[DATA]** and explorations **[EXPLORE]** required to assess the feasibility of our proposals.

??? quote "Our task"

    What are the signals?

    - title, description and other relevant metadata from the content **[DATA]**
    - are the tokens in the content metadata enough signal to identify explicit and implicit tags **[EXPLORE]**

#### Constraints

Discuss the constraints that we have to account for in our solutions. A large majority of constraints can directly come from our service-level agreements (SLAs) with customers and internal systems regarding time, $, performance, latency, infrastructure, privacy, security, UI/UX.

??? quote "Our task"

    - maintain low latency (>100ms) when providing our generated tags. **[Latency]**
    - only recommend tags from our list of approved tags. **[Security]**
    - avoid overwhelming the author with too many predicted tags. **[UI/UX]**

!!! question "Solutions → Constraints or vice versa?"
    Is it naive to consider solutions before constraints because they dictate the nuances of our constraints right?

    ??? quote "Show answer"
        We believe that freely brainstorming solutions without being biased by constraints can lead to very creative solutions. Additionally, in future releases, constraints can often be overcome if the solution motivates it. However, for this current release, it's good to scope the solution by accounting for the constraints. But because we've documented our ideal solution, we can work towards that in future releases.

#### Integration

What are the dependencies and consumers we need to integrate with? Our project team will use this to request comments from appropriate team members and allocate resources for the releases. It's also important to think about coexistence and potential conflicts with other system components as well.

??? quote "Our task"

    - **dependencies**:
        - labeled dataset to benchmark current and subsequent approaches.
        - cluster resources to maintain and scale microservice based on demand.
    - **consumers**:
        - content creation/update UI to consume and display predicted tags. *[MOCK]*

!!! tip "Be transparent"
    If our dependencies are already established, it's important that we let the respective teams know that we are consumers so that we are included in conversations around future changes that may potentially break our system.

### Requirements

Describe core requirements that will help shape the functionality for this specific release. The project teams will use these product requirements to plan the specific [deliverables](#deliverables) to fulfill each requirement.

??? quote "Our task"

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
        <td>must-have</td>
        <td>v1</td>
        <td>Complete</td>
    </tr>
    <tr>
        <td>Predict explicit and implicit tags for given content metadata.</td>
        <td>must-have</td>
        <td>v1</td>
        <td>Complete</td>
    </tr>
    <tr>
        <td>UI to poll prediction service when relevant content metadata is changed during content creation/update process.</td>
        <td>must-have</td>
        <td>v1</td>
        <td>Complete</td>
    </tr>
    <tr>
        <td>Allow authors to click on the recommended tag to automatically add it to their content.</td>
        <td>should-have</td>
        <td>v1</td>
        <td>In-progress</td>
    </tr>
    <tr>
        <td>Remove predicted tags if the author has already manually included them.</td>
        <td>nice-to-have</td>
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

#### Out of scope

What aspects of the feature/problem should we not be concerned with for the immediate planning? Out of scope doesn't mean that we will never address but just not during this specific deliverable.

??? quote "Our task"

    - identify relevant tags beyond our [approved list](https://github.com/GokuMohandas/MadeWithML/blob/main/datasets/tags.json){:target="_blank"} of tags
    - using text from content metadata besides title and description, such as full-text HTML from associated links.
    - interpretability for why we recommend certain tags.

#### Decisions

Given the feasibility evaluation, constraints and what's out of scope, what are the key decisions that need to be made? A recommended framework to use is driver, approver, contributors and informed ([DACI](https://en.wikipedia.org/wiki/Responsibility_assignment_matrix#DACI){:target="_blank"}) responsibility assignment matrix.

??? quote "Our task"
    <table>
    <thead>
    <tr>
        <th>Decision</th>
        <th>Driver</th>
        <th>Approver</th>
        <th>Contributors</th>
        <th>Informed</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>Using only the title and description as relevant signal for recommending tags</td>
        <td>Product DRI</td>
        <td>Senior manager</td>
        <td>Data scientist</td>
        <td>Executives, external stakeholders, etc.</td>
    </tr>
    <tr>
        <td>Recommend tags from our approved list of tags only</td>
        <td>Security team</td>
        <td>Product DRI</td>
        <td>Product DRI</td>
        <td>Executives, external stakeholders, etc.</td>
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

#### Concerns

What are potential risks, concerns and uncertainties that every one should be aware of?

??? quote "Our task"

    - how long to continue to QA every single incoming sample
    - system ways to account for popular tags that are not in our accepted list of tags that we're using to recommend relevant tags

## System design

[*How*]: describe our systemic approach towards building the product.

### Data

Describe the static (ex. labeled dataset) and dynamic (ex. data streams that deliver live features) sources of data that we depend on. This is also the space to think about data privacy/security and sources of data that we *would* like to have but are not available. We can also use this section to describe the process of labeling our dataset and even how we settled on the inputs and target space.

??? quote "Our task"

    - **static**:
        - access to a labeled and validated dataset with content metadata and all relevant tags.
        - information on feature origins and schemas.
        - how did we settle on these labels?
        - what kind of validation did this data go through?
        - was there sampling of any kind applied to create this dataset?
        - are we introducing any data leaks?
    - **dynamic**:
        - access to live content metadata as an author is creating content.

### Evaluation

Regardless of our approach, we need to be able to evaluate how we’re performing. This evaluation criteria needs to be comprehensive and mimic what we can expect in production as much as possible.  This is also the section where we would discuss [offline](evaluation.md){:target="_blank"} and [online](monitoring.md#performance){:target="_blank"} evaluation methods.

#### Metrics

One of the hardest challenges with evaluation is trying the metrics from the product's [objectives](#objectives) with metrics that our model is able to produce. We need to prove that what we're optimizing for is the best direct measure of the core business metric(s) that we're concerned with.

??? quote "Our task"
    We want to be able to suggest highly relevant tags (precision) so we don't fatigue the user with noise. But *recall* that the whole point of this task is to suggest tags that the author will miss (recall) so we can allow our users to find the best resource! We may also want to evaluate performance for specific classes or [slices](evaluation.md#slices){:target="_blank"} of data.

!!! question "What are our priorities"
    For our respective industries or areas of interest, do you we where the priorities are (metrics, errors and other tradeoffs)?

    ??? quote "Show answer"
        It entirely depends on the specific task. For example, in an email spam detector, precision is very important because it's better than we some spam then completely miss an important email. Overtime, we need to iterate on our solution so all evaluation metrics improve but it's important to know which one's we can't comprise on from the get-go.

#### Offline vs. online
For [offline evaluation](evaluation.md){:target="_blank"}, we'll need a gold standard labeled dataset that we can use to benchmark all of our [methods](#methodologies), while for [online evaluation](monitoring.md#performance){:target="_blank"} we can initially use the [proxy signal](monitoring.md#performance){:target="_blank"} on whether the author used our suggested tags and then we can adjust the labels after that new data goes through a proper QA pipeline.

### Methodologies

Describe the different methodologies that each deliver end-to-end utility.

- **End-to-end utility**: the end result from every iteration should deliver minimum end-to-end utility so that we can benchmark iterations against each other and plug-and-play with the system.
- **Keep it simple (KISS)**: start from the simplest solution and slowly add complexity with justification along the way → [baselines](https://madewithml.com/courses/mlops/baselines){:target="_blank"}.
- **Manual before ML**: incorporate deterministic components where we define the rules before using probabilistic ones that infer rules from data.
- **Augment vs. automate**: allow the system to supplement the decision making process as opposed to making the final decision.
- **Internal vs. external**: not all early releases have to be end-user facing. We can use early versions for internal validation, feedback, data collection, etc.
- **Thorough**: every approach needs to be well [tested](testing.md){:target="_blank"} (code, data + models) and [evaluated](evaluation.md){:target="_blank"}, so we can objectively benchmark different approaches.

For the purpose of this course, we're going to develop a solution that involves machine learning from the very beginning. However, we would've followed an iterative approach where we start with simple solutions and slowly add complexity. We would also be evaluating each approach using a gold-standard labeled dataset thats representative of the production space.

??? quote "Our task"

    - v1: creating a gold-standard labeled dataset that is representative of the production space to evaluate approaches with.
    - v2: simple UI change to encourage authors to add specific classes of tags (`frameworks`, `tasks`, `algorithms`, etc.)
    - v3: rule-based approaches using text matching from list of curated tags/aliases
    - v4: predict relevant tags from content title and descriptions
    - v5: ...

!!! warning "Decouple POCs and implementations"
    Each of these approaches would involve proof-of-concept (POC) release and an implementation release after validating it's utility over previous approaches. We should decouple POCs and implementations because if a POC doesn't prove successful, then we can't do the implementation and all the associated planning is no longer applicable.


!!! question "Utility in starting simple"
    Some of the earlier, simpler, approaches may not deliver on a certain performance objective. What are some advantages of still starting simple?

    ??? quote "Show answer"
        - get internal feedback on end-to-end utility
        - perform A/B testing to understand UI/UX design
        - deployed locally to start generating more data required for more complex approaches

#### ML systems

With ML, we’re not writing explicit rules to apply to data but rather using data to learn implicit rules. This inherently involves more trial-and-error compared to composing deterministic systems. Therefore, it’s important to iteratively and regularly scope out the problem towards promising techniques.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/iteration/development_cycle.png" width="700" alt="product development cycle">
</div>

1. Proof of concept (POC) with a static, [labeled](labeling.md) dataset to explore if the input features we have are sufficient enough for the task. Though this is just [baselining](baselines.md){:target="_blank"}, this approach still requires thorough [testing](testing.md){:target="_blank"} (code, data + models) and [evaluation](evaluation.md){:target="_blank"}.
2. [Optimizing](optimization.md){:target="_blank"} on the solution, while potentially using the POC to collect more data, so that the system achieves performance requirements.
3. [Deploy](infrastructure.md){:target="_blank"}, [monitor](monitoring.md){:target="_blank"} and maintain the versioned and reproducible models.
4. If using an end-to-end system, start to decouple into individual [pipeline workflows](pipelines.md){:target="_blank"} that can be scaled, debugged and executed separately. This can involve using constructs such as [feature stores](feature-store.md){:target="_blank"} and [model servers](cicd.md#serving){:target="_blank"} to quickly iterate towards a [continual learning system](continual-learning.md){:target="_blank"} with [data-centric views](data-centric-ai.md){:target="_blank"}.

!!! warning "Always return to the purpose"
    While it's important to iterate and optimize the internals of our workflows, it's even more important to ensure that our ML systems are actually making an impact. We need to constantly engage with stakeholders (management, users) to iterate on why our ML system exists.

#### Rollout

What do the release strategies look like for our different versions? Note that not all releases have to be high stakes, external facing to the whole world. We can always include internal releases, gather feedback and iterate until we’re ready to increase the scope.

- Canary internal rollout, monitoring for proxy/actual performance, etc.
- Rollout to the larger internal team for more feedback.
- A/B rollout to a subset of the population to better understand UX, utility, etc.

#### Feedback

How do we receive feedback on our system and incorporate it into the next iteration? This can involve both human-in-the-loop feedback as well as automatic feedback via [monitoring](monitoring.md){:target="_blank"}, etc.

??? quote "Our task"

    - use feedback from the internal rollout to figure out how to educate early users of the feature with new session hooks.
    - use author's chosen tags as a proxy signal to quantify online performance.
    - enforce human-in-loop checks on where they are conflicts between recommended tags and manually chosen tags.
    - allow users to report issues related to suggested tags.

## Project

[*Who* & *When*]: organizing all the product requirements into manageable timelines so we can deliver on the vision.

### Team

Which teams and specific members from those teams need to be involved in this project? It’s important to consider even the minor features so that everyone is aware of it and so we can properly scope and prioritize our timelines. Keep in mind that this isn’t the only project that people might be working on.

??? quote "Our task"

    - **Product**: the members responsible for outlining the product requirements and approving them may involve product managers, executives, external stakeholders, etc.
    - **System design**:
        - **Data engineering**: these developers are often responsible for the data dependencies, which include robust workflows to continually deliver the data and ensuring that it’s properly validated and ready for downstream applications
        - **Machine learning**: develop the probabilistic systems with appropriate evaluation.
        - **DevOps**: deploy the application and help autoscale based on traffic.
        - **UI/UX**: consume the system’s outputs to deliver the new experience to the user.
        - **Accessibility**: help educate the community for the new rollouts and to assist with decisions around sensitive issues.
        - **Site reliability**: maintain the application and to potentially oversee that online evaluation/monitoring workflows are working as they should.
    - **Project**: the members responsible for iterative engagement with the product and engineering teams to ensure that the right product is being built and that it’s being built appropriately may include project managers, engineering managers, etc.

### Deliverables

We need to break down all the [requirements](#requirements) for a particular release into clear deliverables that specify the deliverable, contributors, dependencies, acceptance criteria and status. This will become the granular checklist that our teams will use to decide what to work on next and to ensure that they’re working on it properly (with all considerations).

??? quote "Our task"
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

### Timeline

This is where the project scoping begins to take place. Often, the stakeholders will have a desired time to release and the functionality to be delivered. There  *will* be a lot of batch and forth on this based on the results from the feasibility studies, so it's very important to be thorough and transparent to set expectations quickly.

??? quote "Our task"
    **v1**: predict relevant tags using content title and description metadata

    - Exploration studies conducted by XX
    - Pushed to dev for A/B testing by XX
    - Pushed to staging with on-boarding hooks by XX
    - Pushed to prod by XX

    This is an extremely simplified timeline. An actual timeline would depict timelines from all the different teams stacked on top of each other with vertical lines at specified time-constraints or version releases.

<!-- Citation -->
{% include "cite.md" %}