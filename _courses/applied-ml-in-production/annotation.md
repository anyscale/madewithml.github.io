---
layout: page
title: Annotation · Applied ML in Production
description: Preparing and labeling our data for exploration.
image: /static/images/courses/applied-ml-in-production/annotation.png
tags: annotation labeling

course-url: /courses/applied-ml-in-production/
next-lesson-url: /courses/applied-ml-in-production/exploratory-data-analysis/
---

<!-- Header -->
<div class="row">
  <div class="col-md-8 col-6 mr-auto">
    <h1 class="page-title">{{ page.title | split: " · " | first }}</h1>
  </div>
  <div class="col-md-4 col-6">
    <div class="btn-group float-right mb-0" role="group">
      <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i
          class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
    </div>
  </div>
</div>
<hr class="mt-0">

<!-- Video -->
<div class="ai-center-all mt-2">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/Kj_5ZO6nsfk?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div>
<div class="ai-center-all mt-2">
  <small>Accompanying video for this lesson. <a href="https://www.youtube.com/madewithml?sub_confirmation=1" target="_blank">Subscribe</a> for updates!</small>
</div>

<div class="alert info mt-4" role="alert">
  <span style="text-align: left;">
    <i class="fas fa-info-circle mr-1"></i> Connect with the author, <i>Goku Mohandas</i>, on
    <a href="https://twitter.com/GokuMohandas" target="_blank">Twitter</a> and
    <a href="https://www.linkedin.com/in/goku" target="_blank">LinkedIn</a> for
    interactive conversations on upcoming lessons.
  </span>
</div>

<h3><u>Intuition</u></h3>
Annotation is the process of identifying the inputs and outputs that are **worth** modeling (*not* just what could be modeled).
- use objective as a guide to determine the necessary signals
- explore creating new signals (via combining data, collecting new data, etc.)
-  iteratively add more features to control complexity and effort

It's also the phase where we can use our deep understanding of the problem, processes, constraints and domain expertise to:
- enhance using supplementary data
- simplify using constraints

And it isn't just about identifying and labeling our initial dataset but also involves thinking about how to make the annotation process more efficient as our dataset grows.
- who will annotate new (streaming) data
- what tools will be used to accelerate the annotation process
- what workflows will be established to track the annotation process

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [0:00](https://www.youtube.com/watch?v=Kj_5ZO6nsfk&t=0s){:target="_blank"} for a video walkthrough of this section.

<h3><u>Application</u></h3>

- *Datasets*
  - [projects.json](https://raw.githubusercontent.com/madewithml/datasets/main/projects.json){:target="_blank"}: projects with title, description and tags (cleaned by mods).
  - [projects_detailed.json](https://raw.githubusercontent.com/madewithml/datasets/main/projects_detailed.json){:target="_blank"}: projects with full-text details and additional URLs.
  - [tags.json](https://raw.githubusercontent.com/madewithml/datasets/main/tags.json){:target="_blank"}: tags used in dropdown to aid autocompletion.
- *Code*
  - [madewithml/applied-ml-in-production](https://github.com/madewithml/applied-ml-in-production){:target="_blank"}: repository for the code in this course.
  - [tagifai.ipynb](https://github.com/madewithml/applied-ml-in-production/blob/master/notebooks/tagifai.ipynb){:target="_blank"}: notebook we'll be using until we get to scripting.

> We'll have a small GitHub Action that runs on a schedule (cron) to constantly update these datasets over time. We'll learn about how these work when we get to the CI/CD lesson.

Recall that our objective was to augment authors to add the appropriate tags for their project so the community can discover them. So we want to use the metadata provided in each project to determine what the relevant tags are. We'll want to start with the highly influential features and iteratively experiment with additional features:
- title + description
- ^ + details
- ^ + relevant html text from URLs

```json
{
  "id": 324,
  "title": "AdverTorch",
  "description": "A Toolbox for Adversarial Robustness Research",
  "tags": [
    "code",
    "library",
    "security",
    "adversarial-learning",
    "adversarial-attacks",
    "adversarial-perturbations"
  ]
}
```

The reason we want to iteratively add more features is because it introduces more complexity and effort. For example, extracting the relevant HTML from the URLs is not trivial but recall that we want to *close the loop* with a simple solution first. We're going to use just the title and description because we hypothesize that the project's core concepts will be there whereas the details may have many other keywords.

We're also going to be using a [supplementary dataset](https://raw.githubusercontent.com/madewithml/datasets/main/tags.json) which contains a collection of all the tags with their aliases and parent/child relationships.
```json
"question-answering": {
    "aliases": [
      "qa"
    ],
    "parents": [
      "natural-language-processing"
    ]
  }
```
We're going to *include* only these tags because they're the tags we care about and we've allowed authors to add any tag they want (noise). We'll also be *excluding* some general tags because they are automatically added when their children tags are present.
```python
# Inclusion/exclusion criteria for tags
include = list(tags_dict.keys())
exclude = ['machine-learning', 'deep-learning',  'data-science',
           'neural-networks', 'python', 'r', 'visualization',
           'natural-language-processing', 'computer-vision']
```
> Keep in mind that because we're *constraining* the output space here, we'll want to monitor the prevalence of new tags over time so we can capture them.

We're also going to restrict the mapping to only tags that are above a certain frequency threshold. The tags that don't have enough projects will not have enough samples to model their relationships.
```python
Most popular tags:
 [('pytorch', 258), ('tensorflow', 213), ('transformers', 196), ('attention', 120), ('convolutional-neural-networks', 106)]

Tags that just made the cut:
 [('time-series', 34), ('flask', 34), ('node-classification', 33), ('question-answering', 32), ('pretraining', 30)]

Tags that just missed the cut:
 [('model-compression', 29), ('fastai', 29), ('graph-classification', 29), ('recurrent-neural-networks', 28), ('adversarial-learning', 28)]
 ```

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [8:14](https://www.youtube.com/watch?v=Kj_5ZO6nsfk&t=494s){:target="_blank"} to see what all of this looks like in [code](https://github.com/madewithml/applied-ml-in-production/blob/master/notebooks/tagifai.ipynb){:target="_blank"}.

Over time, our dataset will grow and we'll need to label new data. So far, we had a team of moderators clean the existing data but we'll need to establish proper workflow to make this process easier and reliable. Typically, we'll use collaborative UIs where annotators can fix errors, etc. and then use a tool like [Airflow](https://airflow.apache.org/) for workflow management to know when new data is ready to be annotated and also when it's ready to be used for modeling.

> In the next section we'll be performing exploratory data analysis (EDA) on our labeled dataset. However, the order of the `annotation` and `EDA` steps can be reversed depending on how well the problem is defined. If you're unsure about what inputs and outputs are worth mapping, use can use EDA to figure it out.

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [2:50](https://www.youtube.com/watch?v=Kj_5ZO6nsfk&t=170s){:target="_blank"} for a video walkthrough of this section.

<h3><u>Resources</u></h3>
- [Human in the Loop: Deep Learning without Wasteful Labelling](https://oatml.cs.ox.ac.uk/blog/2019/06/24/batchbald.html)
- [Harnessing Organizational Knowledge for Machine Learning](https://ai.googleblog.com/2019/03/harnessing-organizational-knowledge-for.html)

<!-- Footer -->
<hr>
<div class="row mb-4">
  <div class="col-6 mr-auto">
    <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
  </div>
  <div class="col-6">
    <div class="float-right">
      <a href="{{ page.next-lesson-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-right mr-1"></i>Next lesson</a>
    </div>
  </div>
</div>