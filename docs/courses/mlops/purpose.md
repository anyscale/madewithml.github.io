---
template: lesson.html
title: Designing Machine Learning Products
description: A template to guide the development cycle for machine learning systems that factors in product requirements, design docs and project considerations.
keywords: project management, product management, design docs, scoping, management, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
---

{% include "styles/lesson.md" %}

## Overview

In this course, we'll not only develop the machine learning models but talk about all the important ML system and software design components required to put our models into production in a reproducible, reliable and robust manner. We'll start by setting the scene for the precise product we'll be building. While this is a technical course, this initial product design process is extremely crucial and is what separates great products from mediocre ones. This lesson will offer the structure for how to think about ML + product.

## Template

This template is designed to guide machine learning product development. While this template will initially be completed in sequential order, it will naturally involve nonlinear engagement based on iterative feedback. We should follow this template for every major release of our products so that all the decision making is transparent and documented.

[Product](#product) (*What* & *Why*) → [System design](#system-design) (*How*) → [Project](#project) (*Who* & *When*)

All great products improve over time and they require release documentation with clear objectives. In this lesson, we'll be designing the initial release but you would do the same for all future releases as well. It's important that we organize our projects this way so that there is one central location where all documentation can be found and kept up-to-date. This also enforces that we scope releases appropriately and provide the thorough planning and documentation to execute each one.

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

!!! note "Assumptions"
    Throughout this lesson, we'll state and justify the assumptions we made to simplify the problem space.

## Product Management

[*What* & *Why*]: motivate the need for the product and outline the objectives and key results.

!!! note
    Each section below has a dropdown component called "Our task", which will discuss the specific topic with respect to the specific product that we're trying to build.

### Overview

#### Background

Describe the problem and product features at a high level. This is the section where you’re setting the scene for someone new, so avoid getting into the details until you reach the sections further below.

??? quote "Our task"

    With so much content on machine learning these days, it's hard to keep things organized. We have (hypothetically) created a platform where people can add and categorize ML content they created or found online for others to discover. However, this process is severely limited by the individuals adding content while our passive consumers want to see a lot more organized content. We do see relevant ML content all over the internet but it's all scattered and unorganized.

    ```json linenums="1" title="Sample data point"
    {
        "id": 443,
        "created_on": "2020-04-10 17:51:39",
        "title": "AllenNLP Interpret",
        "description": "A Framework for Explaining Predictions of NLP Models",
        "tag": "natural-language-processing"
    }
    ```

#### Relevance

Why is this important to work on and why now? We need to justify the efforts for the potential value proposition. As a general rule, it’s good to be as specific as possible in this section and use numerical values to strengthen claims.

??? quote "Our task"

    **Core business values**
    We want to be able to organize the overwhelming amount of ML content and provide a discovery hub for the community.

    **Engagement**
    When our users are able to discover the precise resources for their needs, this drives engagement on our platform and improves perceived value.

#### Objectives

What are the key objectives that we're trying to satisfy (specific success criteria/metrics)?

??? quote "Our task"

    - Increase the amount of daily content on the platform by >20%.
    - Correctly categorize content with >85% precision.

!!! warning "Objectives vs. constraints"
    Objectives and [constraints](#constraints) are often incorrectly used interchangeably but they're very different. Objectives are things that we want to **achieve** that are under our control. Usually, they start with the phrase "We want &lt;objective&gt;" whereas constraints are **limitations** that we need to abide by and often look like "We can't do X because &lt;constraint&gt;". Another way to think about constraints is that it's something we wouldn't impose on ourselves if we didn't have to.

### Solutions

#### Current

Describe current approaches and alternative solutions that were considered and why they're not ideal

??? quote "Our task"

    - Current approach → individuals add content to the platform on a daily basis.
    - Alternatives → increasing marketing efforts to attract more users to add and organize content.
    - Not ideal → manually adding content is the main bottleneck so we need to address this from a different angle.

#### Proposal

Describe the proposed solution and it's features. These solutions may also require separate documentation including wireframes, user stories, mock-ups, etc.

??? quote "Our task"

    We want to create a product that will automatically discover and classify content so that everything is organized for discovery. To simplify our task, let's assume we already have a pipeline that delivers ML content from popular sources (Reddit, Twitter, etc.) as an hourly stream. We can develop a model that can classify the incoming content so that it adheres to how our platform organizes content. From the consumer's point of view, they will (magically) see content on the platform that is updated regularly and organized by category.

#### Feasibility

How realistic is our proposed solution and what are some of the dependencies (ex. data)? We may also discover questions that need further exploration.

??? quote "Our task"

    We have access to all the [projects](https://github.com/GokuMohandas/MadeWithML/blob/main/datasets/projects.json){:target="_blank"} that have been added to the platform already. They have the relevant signals (text) and the respective tag (labels). We can use this historical data to develop a model to replicate this process at scale.

    - data → title, description and other relevant metadata.
    - explore → do the text feature provide adequate signal for the classification task?

#### Constraints

What are the constraints that we have to account for in our solutions? A large majority of constraints can directly come from our service-level agreements (SLAs) with customers and internal systems regarding time, $, performance, latency, infrastructure, privacy, security, UI/UX, etc.

??? quote "Our task"

    - maintain low latency (>100ms) when classifying incoming content. **[Latency]**
    - only recommend tags from our list of approved tags. **[Security]**
    - avoid duplicate content from being added to the platform. **[UI/UX]**

!!! question "Solutions → Constraints or vice versa?"
    Is it naive to consider solutions before constraints because they dictate the nuances of our constraints right?

    ??? quote "Show answer"
        We believe that freely brainstorming solutions without being biased by constraints can lead to very creative solutions. Additionally, in future releases, constraints can often be overcome if the solution motivates it. However, for this current release, it's good to scope the solution by accounting for the constraints. But because we've documented our ideal solution, we can work towards that in future releases.

#### Integration

What are the dependencies and dependents we need to integrate with? Any potential conflicts?

??? quote "Our task"

    - **dependencies**:
        - labeled dataset to benchmark current and future approaches.
        - stream of content from trusted sources (Reddit, Twitter, etc.)
        - resources to run and scale service.
    - **dependents**:
        - platform UI to display categorized content.

### Requirements

Describe core requirements that will help shape the functionality for this specific release. The project team will use these product requirements to plan the specific [deliverables](#deliverables) to fulfill each requirement.

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
        <td>Classify incoming content using provided features.</td>
        <td>must-have</td>
        <td>v1</td>
        <td>Complete</td>
    </tr>
    <tr>
        <td>user feedback process for incorrectly classified content.</td>
        <td>must-have</td>
        <td>v1</td>
        <td>Complete</td>
    </tr>
    <tr>
        <td>Duplicate screening when users add content or via automated discovery pipelines.</td>
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

    - identify relevant tags beyond our approved list of tags.
    - using full-text HTML from content links to aid in classification.
    - interpretability for why we recommend certain tags.
    - identifying multiple relevant categories (see [dataset](#dataset) section for details).

#### Concerns

What are potential risks, concerns and uncertainties that the team should be aware of?

??? quote "Our task"

    - quality assurance on the classifications on incoming data.
    - not capturing new categories that may be highly relevant to our users.

## Systems design

[*How*]: describe our systemic approach towards building the product.

### Data

#### Labeling

Describe the labeling process and how we settled on the features and labels.

??? quote "Our task"

    **Labeling**
    We have [historical data](https://github.com/GokuMohandas/MadeWithML/blob/main/datasets/projects.json){:target="_blank"} that was manually added by users where they categorized content. These labels underwent manual inspection and are of high quality. We also have workflows in place to label and inspect new incoming data so that we can improve our performance.

    **Features**
    We expect the text features (title and description) to offer the signal required for a machine learning model to perform this task.

    **Labels**
    Users were only allowed to add content that belonged to a specific set of categories:

    ```python linenums="1"
    ['natural-language-processing',
     'computer-vision',
     'mlops',
     'reinforcement-learning',
     'graph-learning',
     'time-series',
     'other']
    ```

    <table>
    <thead>
    <tr>
        <th>Assumption</th>
        <th>Actual</th>
        <th>Reason</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>Content can only belong to one category (multiclass).</td>
        <td>Content can belong to more than one category (multilabel).</td>
        <td>Simplicity and many Python libraries don't support multilabel scenarios.</td>
    </tr>
    </tbody>
    </table>

#### Sources

Describe the static (ex. labeled dataset) and dynamic (ex. data streams that deliver live features) sources of data that we depend on.

??? quote "Our task"

    - **static**:
        - access to a labeled and validated datasets for training.
        - information on feature origins and schemas.
        - how did we settle on these labels?
        - what kind of validation did this data go through?
        - was there sampling of any kind applied to create this dataset?
        - are we introducing any data leaks?
    - **dynamic**:
        - access to streams of ML content from scattered sources (Reddit, Twitter, etc.)
        - how can we trust that this stream only has data that is consistent with what we have historically seen?

    <table>
    <thead>
    <tr>
        <th>Assumption</th>
        <th>Actual</th>
        <th>Reason</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>ML stream only has ML relevant content.</td>
        <td>Filter to remove spam content from these 3rd party streams.</td>
        <td>Would require us to source relevant data and build another model.</td>
    </tr>
    </tbody>
    </table>

#### Storage

Describe the [data management systems](labeling.md#data-management-systems){:target="_blank"} (databases, warehouse, [feature store](feature-store.md){:target="_blank"}, etc.) used to store the data.

??? quote "Our task"

    Our training data lives in the form of a [versioned](versioning.md){:target="_blank"} JSON file fetched from a DB and our inference data can come from batch ETL jobs that have aggregating relevant ML content that needs to be classified from across the internet.

### Evaluation

Before we can model our objective, we need to be able to evaluate how we’re performing. This evaluation criteria needs to represent the aspects of the scenario we prioritize.

#### Metrics

One of the hardest challenges with evaluation is trying the metrics from the product's [objectives](#objectives) with metrics that our model is able to produce. We need to prove that what we're optimizing for is the best direct measure of the core business metric(s) that we're concerned with.

??? quote "Our task"
    We want to be able to categorize incoming data with high precision so that we can categorize them under the appropriate topic. For the projects that we categorize as `other`, we can *recall* any misclassified content using manual labeling workflows. We may also want to evaluate performance for specific classes or [slices](evaluation.md#slices){:target="_blank"} of data.

!!! question "What are our priorities"
    For our respective industries or areas of interest, do you we where the priorities are (metrics, errors and other tradeoffs)?

    ??? quote "Show answer"
        It entirely depends on the specific task. For example, in an email spam detector, precision is very important because it's better than we some spam then completely miss an important email. Overtime, we need to iterate on our solution so all evaluation metrics improve but it's important to know which one's we can't comprise on from the get-go.

#### Offline

[Offline evaluation](evaluation.md){:target="_blank"} requires a gold standard labeled dataset that we can use to benchmark all of our [methods](#methodologies).

??? quote "Our task"

    We'll be using the [historical dataset](https://github.com/GokuMohandas/MadeWithML/blob/main/datasets/projects.json){:target="_blank"} for offline evaluation. We'll also be creating [slices](evaluation.md#slices){:target="_blank"} of data that we want to evaluate in isolation.

#### Online

[Online evaluation](monitoring.md#performance){:target="_blank"} can be performed using labels or, in the event we don't readily have labels, [proxy signals](monitoring.md#performance){:target="_blank"}.

??? quote "Out task"

    Proxy signals in our task can be:

    - send manually added content (manually categorized) to compare with the model's prediction.
    - asking the initial set of users viewing a newly categorized content if it's appropriately categorized.
    - allow users to report misclassified content by our model.

### Methodologies

#### Principles

While the specific methodology we employ can differ based on the problem, there are core principles we always want to follow:

- **End-to-end utility**: the end result from every iteration should deliver minimum end-to-end utility so that we can benchmark iterations against each other and plug-and-play with the system.
- **Manual before ML**: incorporate deterministic components where we define the rules before using probabilistic ones that infer rules from data → [baselines](https://madewithml.com/courses/mlops/baselines){:target="_blank"}.
- **Augment vs. automate**: allow the system to supplement the decision making process as opposed to making the final decision.
- **Internal vs. external**: not all early releases have to be end-user facing. We can use early versions for internal validation, feedback, data collection, etc.
- **Thorough**: every approach needs to be well [tested](testing.md){:target="_blank"} (code, data + models) and [evaluated](evaluation.md){:target="_blank"}, so we can objectively benchmark different approaches.

??? quote "Our task"

    - v1: creating a gold-standard labeled dataset that is representative of the problem space.
    - v2: rule-based text matching approaches to categorize content.
    - v3: probabilistically predicting labels from content title and description.
    - v4: ...

    <table>
    <thead>
    <tr>
        <th>Assumption</th>
        <th>Actual</th>
        <th>Reason</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>Solution needs to involve ML due to unstructured data and open domain space.</td>
        <td>An iterative approach where we start with simple rule-based solutions and slowly add complexity.</td>
        <td>This course is about responsibly deliver value with ML, so we'll jump to it right away.</td>
    </tr>
    </tbody>
    </table>

!!! warning "Decouple POCs and implementations"
    Each of these approaches would involve proof-of-concept (POC) release and an implementation release after validating it's utility over previous approaches. We should decouple POCs and implementations because if a POC doesn't prove successful, then we can't do the implementation and all the associated planning is no longer applicable.

!!! question "Utility in starting simple"
    Some of the earlier, simpler, approaches may not deliver on a certain performance objective. What are some advantages of still starting simple?

    ??? quote "Show answer"
        - get internal feedback on end-to-end utility.
        - perform A/B testing to understand UI/UX design.
        - deployed locally to start generating more data required for more complex approaches.

#### Maturation

As we prove out our proof of concepts (POCs), we'll want make our system increasingly mature so that less manual intervention is required.

??? quote "Out task"

    1. POC with a static, [labeled](labeling.md) dataset to explore if the input features we have are sufficient enough for the task. Though this is just [baselining](baselines.md){:target="_blank"}, this approach still requires thorough [testing](testing.md){:target="_blank"} (code, data + models) and [evaluation](evaluation.md){:target="_blank"}.
    2. [Optimizing](optimization.md){:target="_blank"} on the solution, while potentially using the POC to collect more data, so that the system achieves performance requirements.
    3. [Deploy](infrastructure.md){:target="_blank"}, [monitor](monitoring.md){:target="_blank"} and maintain the versioned and reproducible models.
    4. If using an end-to-end system, start to decouple into individual [pipeline workflows](pipelines.md){:target="_blank"} that can be scaled, debugged and executed separately. This can involve using constructs such as [feature stores](feature-store.md){:target="_blank"} and [model servers](cicd.md#serving){:target="_blank"} to quickly iterate towards a [continual learning system](continual-learning.md){:target="_blank"}.

!!! warning "Always return to the value proposition"
    While it's important to iterate and optimize the internals of our workflows, it's even more important to ensure that our ML systems are actually making an impact. We need to constantly engage with stakeholders (management, users) to iterate on why our ML system exists.

    <div class="ai-center-all">
        <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/iteration/development_cycle.png" width="600" alt="product development cycle">
    </div>

#### Rollout

What do the [online evaluation](evaluation.md#online-evaluation){:target="_blank"} (experimentation) and release strategies look like for our different versions? Note that not all releases have to be high stakes, external facing to the whole world. We can always include internal releases, gather feedback and iterate until we’re ready to increase the scope.

- Canary internal rollout, monitoring for proxy/actual performance, etc.
- Rollout to the larger internal team for more feedback.
- A/B rollout to a subset of the population to better understand UX, utility, etc.

#### Feedback

How do we receive feedback on our system and incorporate it into the next iteration? This can involve both human-in-the-loop feedback as well as automatic feedback via [monitoring](monitoring.md){:target="_blank"}, etc.

??? quote "Our task"

    - use author's chosen tags as a proxy signal to quantify online performance.
    - enforce human-in-loop checks when there is low confidence in classifications.
    - allow users to report issues related to misclassification.

## Project Management

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