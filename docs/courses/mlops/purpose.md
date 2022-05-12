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

While our documentation will be detailed, we can start the process by walking through a machine learning canvas:

<div class="ai-center-all">
    <a href="/static/templates/ml-canvas.pdf" target="_blank"><img src="/static/images/mlops/purpose/ml_canvas.png" width="1000" alt="machine learning canvas"></a>
</div>

👉 &nbsp; Download a PDF of the ML canvas to use for your own products → [ml-canvas.pdf](/static/templates/ml-canvas.pdf){:target="_blank"} (right click the link and hit "Save Link As...")

From this high-level canvas, we can create detailed documentation for each release:

```bash
# Documentation
📂 project/
├── 📄 Overview
├── 📂 release-1
| ├── 📄 product requirements [Product]
| ├── 📄 design documentation [System design]
| ├── 📄 project planning     [Project]
├── ...
└── 📂 release-n
```

> Throughout this lesson, we'll state and justify the assumptions we made to simplify the problem space.

## Product Management

[*What* & *Why*]: motivate the need for the product and outline the objectives and key results.

!!! note
    Each section below has a dropdown component called "Our task", which will discuss the specific topic with respect to the specific product that we're trying to build.

### Overview

#### Background

Set the scene for what we're trying to do through a customer-centric approach:

- `#!js customer`: profile of the customer we want to address
- `#!js goal`: main goal for the customer
- `#!js pains`: obstacles in the way of the customer achieving the goal
- `#!js gains`: what would make the job easier for the customer?

??? quote "Our task"

    - `#!js customer`: machine learning developers and researchers.
    - `#!js goal`: stay up-to-date on ML content for work, knowledge, etc.
    - `#!js pains`: too much uncategorized content scattered around the internet.
    - `#!js gains`: a central location with categorized content from trusted 3rd party sources.

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
        <td>Assume we have customers on our platform already.</td>
        <td>Doesn't matter what you build if no one is there to use it.</td>
        <td>This is a course on ML, not cold start growth.</td>
    </tr>
    <tr>
        <td>Customers want to stay up-to-date with ML content.</td>
        <td>Thorough customer studies required to confirm this.</td>
        <td>We need a task that we can model in this course.</td>
    </tr>
    </tbody>
    </table>

#### Value proposition

Propose the value we can create through a product-centric approach:

- `#!js product`: what needs to be build to help the customer reach their goal
- `#!js alleviates`: how will the product reduce pains?
- `#!js advantages`: how will the product create gains?

??? quote "Our task"

    - `#!js product`: service that discovers and categorizes ML content from popular sources.
    - `#!js alleviates`: timely display categorized content for customers to discover.
    - `#!js advantages`: customers only have to visit our product to stay up-to-date.

#### Objectives

Breakdown the product into key objectives that we want to focus on.

??? quote "Our task"

    - Allow customers to add and categorize their own projects.
    - Discover ML content from trusted sources to bring into our platform.
    - Classify incoming content (>85% precision) for our customers to easily discover. **[OUR FOCUS]**
    - Display categorized content on our platform (recent, popular, recommended, etc.)

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
        <td>Assume we have a pipeline that delivers ML content from popular sources (Reddit, Twitter, etc.).</td>
        <td>We would have to build this as a batch service and is not trivial.</td>
        <td>This is a course on ML, not batch web scraping.</td>
    </tr>
    </tbody>
    </table>

### Solution

Describe the solution required to meet our objectives, including it's core features, integration, alternatives, constraints and what's out-of-scope.

> May require separate documentation (wireframes, user stories, mock-ups, etc.).

??? quote "Our task"

    Develop a model that can classify the incoming content so that it can be organized by category on our platform.

    `#!js Core features`:

    - ML service that will predict the correct categories for incoming content. **[OUR FOCUS]**
    - user feedback process for incorrectly classified content.
    - workflows to categorize content that the service was incorrect about or not as confident in.
    - duplicate screening for content that already exists on the platform.

    `#!js Integrations`:

    - categorized content will be sent to the UI service to be displayed.
    - classification feedback from users will sent to labeling workflows.

    `#!js Alternatives`:

    - allow users to add content manually (bottleneck).

    `#!js Constraints`:

    - maintain low latency (>100ms) when classifying incoming content. **[Latency]**
    - only recommend tags from our list of approved tags. **[Security]**
    - avoid duplicate content from being added to the platform. **[UI/UX]**

    `#!js Out-of-scope`:

    - identify relevant tags beyond our approved list of tags.
    - using full-text HTML from content links to aid in classification.
    - interpretability for why we recommend certain tags.
    - identifying multiple categories (see [dataset](#dataset) section for details).

#### Feasibility

How feasible is our solution and do we have the required resources to deliver it (data, $, team, etc.)?

??? quote "Our task"

    We have a dataset of ML content that have been labeled. We'll need to assess if it has the necessary signals to meet our [objectives](#objectives).

    ```json linenums="1" title="Sample data point"
    {
        "id": 443,
        "created_on": "2020-04-10 17:51:39",
        "title": "AllenNLP Interpret",
        "description": "A Framework for Explaining Predictions of NLP Models",
        "tag": "natural-language-processing"
    }
    ```

## Systems design

[*How*]: describe our systemic approach towards building the product.

### Data

Describe the training and production (batches/streams) sources of data.

??? quote "Our task"

    - **training**:
        - access to a labeled and validated [dataset](https://github.com/GokuMohandas/MadeWithML/blob/main/datasets/projects.json){:target="_blank"} for training.
        - information on feature origins and schemas.
        - was there sampling of any kind applied to create this dataset?
        - are we introducing any data leaks?
    - **production**:
        - access to timely batches of ML content from scattered sources (Reddit, Twitter, etc.)
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

#### Labeling

Describe the labeling process and how we settled on the features and labels.

??? quote "Our task"

    **Labeling**: manually labeled [historical data](https://github.com/GokuMohandas/MadeWithML/blob/main/datasets/projects.json){:target="_blank"}.

    **Features**: text features (title and description) to provide signal for the classification task.

    **Labels**: reflect the content categories we currently display on our platform:

    ```python linenums="1"
    ['natural-language-processing',
     'computer-vision',
     'mlops',
      ...
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
        <td>For simplicity and many libraries don't support or complicate multilabel scenarios.</td>
    </tr>
    </tbody>
    </table>

### Evaluation

Before we can model our objective, we need to be able to evaluate how we’re performing.

#### Metrics

One of the hardest challenges with evaluation is tying our core [objectives](#objectives) (may be qualitative) with quantitative metrics that our model can optimize on.

??? quote "Our task"
    We want to be able to classify incoming data with high precision so we can display them properly. For the projects that we categorize as `other`, we can *recall* any misclassified content using manual labeling workflows. We may also want to evaluate performance for specific classes or [slices](evaluation.md#slices){:target="_blank"} of data.

!!! question "What are our priorities"
    How do we decide which metrics to prioritize?

    ??? quote "Show answer"
        It entirely depends on the specific task. For example, in an email spam detector, precision is very important because it's better than we some spam then completely miss an important email. Overtime, we need to iterate on our solution so all evaluation metrics improve but it's important to know which one's we can't comprise on from the get-go.

#### Offline

[Offline evaluation](evaluation.md){:target="_blank"} requires a gold standard labeled dataset that we can use to benchmark all of our [modeling](#modeling).

??? quote "Our task"

    We'll be using the [historical dataset](https://github.com/GokuMohandas/MadeWithML/blob/main/datasets/projects.json){:target="_blank"} for offline evaluation. We'll also be creating [slices](evaluation.md#slices){:target="_blank"} of data that we want to evaluate in isolation.

#### Online

Online evaluation ensures that our model continues to perform well in production and can be performed using labels or, in the event we don't readily have labels, [proxy signals](monitoring.md#performance){:target="_blank"}.

??? quote "Our task"

    - manually label a subset of incoming data to evaluate periodically.
    - asking the initial set of users viewing a newly categorized content if it's correctly classified.
    - allow users to report misclassified content by our model.

### Modeling

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

#### Experimentation

How can we [experiment](infrastructure.md#experimentation){:target="_blank"} with our model to measure real-time performance before committing to replace our existing version of the system.

> Not all releases have to be high stakes and external facing. We can always include internal releases, gather feedback and iterate until we’re ready to increase the scope.

- Internal canary rollout, monitoring for proxy/actual performance, etc.
- Rollout to the larger internal team for more feedback.
- A/B rollout to a subset of the population to better understand UX, utility, etc.

#### Feedback

How do we receive feedback on our system and incorporate it into the next iteration? This can involve both human-in-the-loop feedback as well as automatic feedback via [monitoring](monitoring.md){:target="_blank"}, etc.

??? quote "Our task"

    - enforce human-in-loop checks when there is low confidence in classifications.
    - allow users to report issues related to misclassification.

!!! warning "Always return to the value proposition"
    While it's important to iterate and optimize the internals of our workflows, it's even more important to ensure that our ML systems are actually making an impact. We need to constantly engage with stakeholders (management, users) to iterate on why our ML system exists.

    <div class="ai-center-all">
        <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/iteration/development_cycle.png" width="600" alt="product development cycle">
    </div>

## Project Management

[*Who* & *When*]: organizing all the product requirements into manageable timelines so we can deliver on the vision.

### Team

Which teams and specific members from those teams need to be involved in this project? It’s important to consider even the minor features so that everyone is aware of it and so we can properly scope and prioritize our timelines. Keep in mind that this isn’t the only project that people might be working on.

??? quote "Our task"

    - **Product**: the members responsible for outlining the product requirements and approving them may involve product managers, executives, external stakeholders, etc.
    - **System design**:
        - **Data engineering**: responsible for the data dependencies, which include robust workflows to continually deliver the data and ensuring that it’s properly validated and ready for downstream applications.
        - **Machine learning**: develop the probabilistic systems with appropriate evaluation.
        - **DevOps**: deploy the application and help autoscale based on traffic.
        - **UI/UX**: consume the system’s outputs to deliver the new experience to the user.
        - **Accessibility**: help educate the community for the new rollouts and to assist with decisions around sensitive issues.
        - **Site reliability**: maintain the application and to potentially oversee that online evaluation/monitoring workflows are working as they should.
    - **Project**: the members responsible for iterative engagement with the product and engineering teams to ensure that the right product is being built and that it’s being built appropriately may include project managers, engineering managers, etc.

### Deliverables

We need to break down all the [objectives](#objectives) for a particular release into clear deliverables that specify the deliverable, contributors, dependencies, acceptance criteria and status. This will become the granular checklist that our teams will use to decide what to prioritize.

??? quote "Our task"
    <table>
    <thead>
    <tr>
        <th>Objective</th>
        <th>Priority</th>
        <th>Release</th>
        <th>Status</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>Classify incoming content (>85% precision) for our customers to easily discover.</td>
        <td>High</td>
        <td>v1</td>
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
        <td>Labeled dataset for training</td>
        <td>Project DRI, labeling team, data engineer</td>
        <td>Access to location of content with relevant metadata</td>
        <td>Validation of ground-truth labels</td>
        <td>Complete</td>
    </tr>
    <tr>
        <td>Trained model with high >85% precision</td>
        <td>Data scientist</td>
        <td>Labeled dataset</td>
        <td>Versioned, reproducible, test coverage report and evaluation results</td>
        <td>In-progress</td>
    </tr>
    <tr>
        <td>Scalable service for inference</td>
        <td>ML engineer, DevOps engineer</td>
        <td>Versioned, reproducible, tested and evaluated model</td>
        <td>Stress tests to ensure autoscaling capabilities</td>
        <td>Pending</td>
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

This is where the project scoping begins to take place. Often, the stakeholders will have a desired time for release and the functionality to be delivered. There *will* be a lot of back and forth on this based on the results from the feasibility studies, so it's very important to be thorough and transparent to set expectations.

??? quote "Our task"
    **v1**: classify incoming content (>85% precision) for our customers to easily discover.

    - Exploration studies conducted by XX
    - Pushed to dev for A/B testing by XX
    - Pushed to staging with on-boarding hooks by XX
    - Pushed to prod by XX

    > This is an extremely simplified timeline. An actual timeline would depict timelines from all the different teams stacked on top of each other with vertical lines at specified time-constraints or version releases.

<!-- Citation -->
{% include "cite.md" %}