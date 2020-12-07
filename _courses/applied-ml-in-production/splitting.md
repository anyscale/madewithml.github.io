---
layout: page
title: Splitting · Applied ML in Production
description: Appropriately splitting our dataset (multi-label) for training, validation and testing.
image: /static/images/courses/applied-ml-in-production/splitting.png
tags: splitting validation stratification

course-url: /courses/applied-ml-in-production/
next-lesson-url: /courses/applied-ml-in-production/preprocessing/
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
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/pKzjkb-M4f0?rel=0" frameborder="0"
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

**Why do we need to split our data?** To determine the efficacy of our models, we need to have an unbiased measuring approach. To do this, we split our dataset into `training`, `validation`, and `testing` data splits. Here is the process:
1. Use the training split to train the model.
  > Here the model will have access to both inputs and outputs to optimize its internal weights.
2. After each loop (epoch) of the training split, we will use the validation split to determine model performance.
  > Here the model will not use the outputs to optimize its weights but instead, we will use the performance to optimize training hyperparameters such as the learning rate, etc.
3. After training stops (epoch(s)), we will use the testing split to perform a one-time assessment of the model.
  > This is our best measure of how the model may behave on new, unseen data. Note that *training stops* when the performance improvement is not significant or any other stopping criteria that we may have specified.

If you are not familiar with some of these terms, be sure to check out any of the following introductory ML content:
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning){:target="_blank"} — Deeplearning AI (Andrew Ng)
- [Intro to ML for Coders](https://course18.fast.ai/ml){:target="_blank"} — Fast AI (Jeremy Howard)
- [Machine Learning Basics](https://github.com/madewithml/basics){:target="_blank"} (among [top 10](https://github.com/topics/deep-learning){:target="_blank"} DL repos on GitHub) — Made With ML (Goku Mohandas)

**How should the data be split?** We need to ensure that our data is properly split so we can trust our evaluations. A few criteria are:
- the dataset (and each data split) should be representative of data we will encounter
- equal distributions of output values across all splits
- shuffle your data if it's organized in a way that prevents input variance
- avoid random shuffles if you task can suffer from data leaks (ex. `time-series`)

> You need to [clean]({% link _courses/applied-ml-in-production/preprocessing.md %}) your data first before splitting, at least for the features that splitting depends on. So the process is more like: preprocessing (global, cleaning) → splitting → preprocessing (local, transformations).

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [0:00](https://www.youtube.com/watch?v=pKzjkb-M4f0&list=PLqy_sIcckLC2jrxQhyqWDhL_9Uwxz8UFq&index=7&t=0s){:target="_blank"} for a video walkthrough of this section.

<h3><u>Application</u></h3>

> The notebook for this section can be found [here](https://github.com/madewithml/applied-ml-in-production/blob/master/notebooks/tagifai.ipynb){:target="_blank"}.

Before we split our dataset, we're going to encode our output labels where we'll be assigning each tag a unique index:

```python
label_encoder.class_to_index = {
  "attention": 0,
  "autoencoders": 1,
  "convolutional-neural-networks": 2,
  "data-augmentation": 3,
  ...
  "transfer-learning": 29,
  "transformers": 30,
  "unsupervised-learning": 31
}
```

and convert each input's list of tags into a one-hot representation.

```python
label_encoder.transform([["attention", "data-augmentation"]])
> [1, 0, 0, 1, 0, ..., 0]
```

For traditional `multi-class` tasks (each input has one label), we want to ensure that each data split has similar class distributions. However, our task is `multi-label` classification (an input can have many labels) which complicates the stratification process.

First, we'll naively split our dataset randomly and show the large deviations between the (adjusted) class distributions across the splits.

<figure>
  <img src="/static/images/courses/applied-ml-in-production/naive_split.png" width="700" alt="pivot">
  <figcaption>Adjusted class distributions after randomly splitting the data.</figcaption>
</figure>

Next we'll apply [iterative stratification](http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf) via the [skmultilearn](http://scikit.ml/index.html) library, which essentially splits each input into subsets (where each label is considered individually) and then it distributes the samples starting with fewest "positive" samples and working up to the inputs that have the most labels.

<figure>
  <img src="/static/images/courses/applied-ml-in-production/iterative_split.png" width="700" alt="pivot">
  <figcaption>Adjusted class distributions after iteratively splitting the data.</figcaption>
</figure>

> We did iterative stratification with `order=1`, but we can account for [higher-order](https://arxiv.org/abs/1704.08756) label relationships as well where we may care about the distribution of label combinations.

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [6:45](https://www.youtube.com/watch?v=pKzjkb-M4f0&list=PLqy_sIcckLC2jrxQhyqWDhL_9Uwxz8UFq&index=7&t=370s){:target="_blank"} for a video walkthrough of this section.

<h3><u>Resources</u></h3>
- [How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/)


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