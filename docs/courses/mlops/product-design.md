---
template: lesson.html
title: Machine Learning Product Design
description: An overview of the machine learning product design process.
keywords: product design, product management, design, design docs, scoping, management, mlops, machine learning
image: https://madewithml.com/static/images/mlops/design/ml_canvas.png
---

{% include "styles/lesson.md" %}

## Overview

Before we start developing any machine learning models, we need to first motivate and design our application. While this is a technical course, this initial product design process is extremely crucial for creating great products. We'll focus on the product design aspects of our application in this lesson and the systems design aspects in the [next lesson](systems-design.md){:target="_blank"}.

## Template

The template below is designed to guide machine learning product development. It involves both the product and systems design ([next lesson](systems-design.md){:target="_blank"}) aspects of our application:

[Product design](product-design.md){:target="_blank"} (*What* & *Why*) → [Systems design](systems-design.md){:target="_blank"} (*How*)

<div class="ai-center-all">
    <a href="/static/templates/ml-canvas.pdf" target="_blank"><img src="/static/images/mlops/design/ml_canvas.png" width="1000" alt="machine learning canvas"></a>
</div>

> 👉 &nbsp; Download a PDF of the ML canvas to use for your own products → [ml-canvas.pdf](/static/templates/ml-canvas.pdf){:target="_blank"} (right click the link and hit "Save Link As...")

## Product design

Motivate the need for the product and outline the objectives and impact.

!!! note
    Each section below has a part called "Our task", which will discuss how the specific topic relates to the application that we will be building.

### Background

Set the scene for what we're trying to do through a user-centric approach:

- `#!js users`: profile/persona of our users
- `#!js goals`: our users' main goals
- `#!js pains`: obstacles preventing our users from achieving their goals

!!! quote "Our task"

    - `#!js users`: machine learning developers and researchers.
    - `#!js goals`: stay up-to-date on ML content for work, knowledge, etc.
    - `#!js pains`: too much unlabeled content scattered around the internet.

### Value proposition

Propose the value we can create through a product-centric approach:

- `#!js product`: what needs to be build to help our users reach their goals?
- `#!js alleviates`: how will the product reduce pains?
- `#!js advantages`: how will the product create gains?

!!! quote "Our task"

    We will build a platform that helps machine learning developers and researchers stay up-to-date on ML content. We'll do this by discovering and categorizing content from popular sources (Reddit, Twitter, etc.) and displaying it on our platform. For simplicity, assume that we already have a pipeline that delivers ML content from popular sources to our platform. We will just focus on developing the ML service that can correctly categorize the content.

    - `#!js product`: a service that discovers and categorizes ML content from popular sources.
    - `#!js alleviates`: display categorized content for users to discover.
    - `#!js advantages`: when users visit our platform to stay up-to-date on ML content, they don't waste time searching for that content themselves in the noisy internet.

    <div class="ai-center-all">
        <img src="/static/images/mlops/design/product.png" width="1000" alt="product mockup"></a>
    </div>

### Objectives

Breakdown the product into key objectives that we want to focus on.

!!! quote "Our task"

    - Discover ML content from trusted sources to bring into our platform.
    - Classify incoming content for our users to easily discover. **[OUR FOCUS]**
    - Display categorized content on our platform (recent, popular, recommended, etc.)

### Solution

Describe the solution required to meet our objectives, including its:

- `#!js core features`: key features that will be developed.
- `#!js integration`: how the product will integrate with other services.
- `#!js alternatives`: alternative solutions that we should considered.
- `#!js constraints`: limitations that we need to be aware of.
- `#!js out-of-scope.`: features that we will not be developing for now.

!!! quote "Our task"

    Develop a model that can classify the content so that it can be organized by category (tag) on our platform.

    `#!js Core features`:

    - predict the correct tag for a given content. **[OUR FOCUS]**
    - user feedback process for incorrectly classified content.
    - workflows to categorize ML content that our model is incorrect / unsure about.

    `#!js Integrations`:

    - ML content from reliable sources will be sent to our service for classification.

    `#!js Alternatives`:

    - allow users to add content manually and classify them (noisy, cold start, etc.)

    `#!js Constraints`:

    - maintain low latency (>100ms) when classifying incoming content. **[Latency]**
    - only recommend tags from our list of approved tags. **[Security]**
    - avoid duplicate content from being added to the platform. **[UI/UX]**

    `#!js Out-of-scope`:

    - identify relevant tags beyond our approved list of tags (`natural-language-processing`, `computer-vision`, `mlops` and `other`).
    - using full-text HTML from content links to aid in classification.

### Feasibility

How feasible is our solution and do we have the required resources to deliver it (data, $, team, etc.)?

!!! quote "Our task"

    We have a [dataset](https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv){:target="_blank"} with ML content that has been labeled. We'll need to assess if it has the necessary signals to meet our [objectives](#objectives).

    ```json linenums="1" title="Sample data point"
    {
        "id": 443,
        "created_on": "2020-04-10 17:51:39",
        "title": "AllenNLP Interpret",
        "description": "A Framework for Explaining Predictions of NLP Models",
        "tag": "natural-language-processing"
    }
    ```

Now that we've set up the product design requirements for our ML service, let's

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}