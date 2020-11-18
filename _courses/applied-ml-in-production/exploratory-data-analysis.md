---
layout: page
title: Exploratory Data Analysis · Applied ML in Production
description: Exploring our dataset for insights with intention.
image: /static/images/courses/applied-ml-in-production/eda.png
tags: matplotlib

course-url: /courses/applied-ml-in-production/
next-lesson-url: /courses/applied-ml-in-production/splitting/
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
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/3Vrlvrjigvs?rel=0" frameborder="0"
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
Exploratory data analysis (EDA) is a vital (and fun) step in the data science process but it's often misconstrued. Here's how to think about EDA:
- not just to visualize a prescribed set of plots (correlation matrix, etc.).
- goal is to *convince* yourself that the data you currently have is sufficient for the task.
- use EDA to answer questions and ask yourself why those questions are important.
- not a one time process; as your data grows, you want to revisit EDA to catch distribution shifts, anomalies, etc.

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [0:00](https://www.youtube.com/watch?v=Kj_5ZO6nsfk&t=0s){:target="_blank"} for a video walkthrough of this section.

<h3><u>Application</u></h3>

> The code for this section can be found [here](https://github.com/madewithml/applied-ml-in-production/blob/master/notebooks/tagifai.ipynb){:target="_blank"}.

**Q1**. How many (post filtered) tags do the projects have? We care about this because we want to make sure we don't overwhelm the user with too many tags (UX constraint).

<figure>
  <img src="/static/images/courses/applied-ml-in-production/eda1.png" width="500" alt="pivot">
  <figcaption>Distribution of tag counts per project</figcaption>
</figure>

**Q2**. What are the most popular tags? We care about this because it's important to know about the distribution of tags and what tags just made the cut (for performance).

<figure>
  <img src="/static/images/courses/applied-ml-in-production/eda2.png" width="600" alt="pivot">
  <figcaption>Distribution of tags across projects</figcaption>
</figure>

**Q3**. Is there enough signal in the title and description that's unique to each tag? This is important because we want to verify our initial hypothesis that the project's title and description are highly influential features.

<figure>
  <img src="/static/images/courses/applied-ml-in-production/eda3.png" width="450" alt="pivot">
  <figcaption>Wordcloud for the tag pytorch</figcaption>
</figure>

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [3:27](https://www.youtube.com/watch?v=3Vrlvrjigvs&t=207s){:target="_blank"} to see what all of this looks like in [code](https://github.com/madewithml/applied-ml-in-production/blob/master/notebooks/tagifai.ipynb){:target="_blank"}.

> All of the work we've done so far are inside IPython notebooks but in a later lesson, we'll transfer all of this into an interactive dashboard using a tool called [Streamlit](https://streamlit.io/){:target="_blank"}.

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [1:37](https://www.youtube.com/watch?v=3Vrlvrjigvs&t=97s){:target="_blank"} for a video walkthrough of this section.

<h3><u>Resources</u></h3>
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/)
- [Data Viz](https://armsp.github.io/covidviz/)

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