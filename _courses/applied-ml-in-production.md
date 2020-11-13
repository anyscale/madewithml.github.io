---
layout: page
title: Applied ML in Production
description: "A guide and code-driven case study on MLOps for software engineers, data scientists and product managers."
image: /static/images/courses/applied-ml-in-production/card.png
redirect_from: /courses/putting-ml-in-production/
tags: applied-ml mlops production

---

<div class="row">
  <div class="col-md-8 col-6 mr-auto">
    <h1 class="page-title">{{ page.title }}</h1>
  </div>
  <div class="col-md-4 col-6">
    <div class="btn-group float-right mb-0" role="group">
      <a href="{% link _courses/index.md %}" class="btn btn-sm btn-outline-secondary"><i
          class="fas fa-sm fa-arrow-left mr-1"></i>Return to courses</a>
    </div>
  </div>
</div>
<hr class="mt-0">

A guide and code-driven case study on MLOps for software engineers, data scientists and product managers. We will be developing an end-to-end ML feature, from product → ML → production, with open source tools.

<div class="alert info" role="alert">
  <span style="text-align: left;">
    <i class="fas fa-info-circle mr-1"></i> Connect with the author, <i>Goku Mohandas</i>, on
    <a href="https://twitter.com/GokuMohandas" target="_blank">Twitter</a> and
    <a href="https://www.linkedin.com/in/goku" target="_blank">LinkedIn</a> for
    interactive conversations on upcoming lessons.
  </span>
</div>

<hr>

<h2><u>Lessons</u></h2>

<div class="row mt-4">
  <div class="col-md-4">
    <b><span class="mr-1">📦</span> Product</b>
    <ul>
      <li><a href="{% link _courses/applied-ml-in-production/objective.md %}">Objective</a></li>
      <li><a href="{% link _courses/applied-ml-in-production/solution.md %}">Solution</a></li>
      <li><a href="{% link _courses/applied-ml-in-production/evaluation.md %}">Evaluation</a></li>
      <li><a href="{% link _courses/applied-ml-in-production/iteration.md %}">Iteration</a></li>
    </ul>
    <b><span class="mr-1">🔢</span> Data</b>
    <ul>
      <li><a href="{% link _courses/applied-ml-in-production/annotation.md %}">Annotation</a></li>
      <li>Exploratory data analysis</li>
      <li>Splitting</li>
      <li>Preprocessing</li>
      <li>Imbalanced data</li>
      <li>Augmentation</li>
      <li>Versioning</li>
    </ul>
    <b><span class="mr-1">🤖</span> Modeling</b>
    <ul>
      <li>Random</li>
      <li>Rule-based</li>
      <li>Sklearn</li>
      <li>CNN</li>
      <li>Transformers</li>
      <li>Interpretability</li>
    </ul>
  </div>
  <div class="col-md-4">
    <b><span class="mr-1">📝</span> Scripting</b>
    <ul>
      <li>OOPs</li>
      <li>Packaging</li>
      <li>Logging</li>
      <li>Testing (basics)</li>
      <li>Testing (ML)</li>
      <li>Formatting</li>
      <li>Makefile</li>
      <li>Precommit</li>
      <li>Git</li>
    </ul>
    <b><span class="mr-1">🎛</span> Tuning</b>
    <ul>
      <li>Experiment tracking</li>
      <li>Optimization</li>
    </ul>
    <b><span class="mr-1">🛠</span> API</b>
    <ul>
      <li>RESTful API</li>
      <li>Databases</li>
      <li>Authentication</li>
      <li>Documentation</li>
    </ul>
  </div>
  <div class="col-md-4">
    <b><span class="mr-1">🚀</span> Production</b>
    <ul>
      <li>Dashboard</li>
      <li>Docker</li>
      <li>Serving</li>
      <li>Monitoring (performance, drift)</li>
      <li>CI/CD (GitHub actions)</li>
      <li>Active learning</li>
      <li>Scaling</li>
    </ul>
  </div>
</div>


<hr>

<h2><u>FAQ</u></h2>

#### Who is this course for?
- ML developers looking to become end-to-end ML developers.
- Software engineers looking to learn how to responsibly deploy and monitor ML systems.
- Product managers who want to have a comprehensive understanding of the different stages of ML dev.

#### What is the structure?
Lessons will be released weekly and each one will include:
- *Intuition*: high level overview of the concepts that will be covered and how it all fits together.
- *Code*: simple code examples to illustrate the concept.
- *Application*: applying the concept to our specific task.
- *Extensions*: brief look at other tools and techniques that will be useful for difference situations.

#### What are the prerequisites?
You should have some familiarity with Python and [basic ML algorithms](https://github.com/madewithml/basics). While we will be experimenting with complex model architectures, you can easily apply the lessons to any class of ML models.

#### What makes this course unique?
- *Hands-on*: If you search production ML or MLOps online, you'll find great blog posts and tweets. But in order to really understand these concepts, you need to implement them. Unfortunately, you don’t see a lot of the inner workings of running production ML because of scale, proprietary content & expensive tools. However, Made With ML is free, open and live which makes it a perfect learning opportunity for the community.
- *Intuition-first*: We will never jump straight to code. In every lesson, we will develop intuition for the concepts and think about it from a product perspective.
- *Software engineering*: This course isn't just about ML. In fact, it's mostly about clean software engineering! We'll cover important concepts like versioning, testing, logging, etc. that really makes this a production-grade product.
- *Focused yet holistic*: For every concept, we'll not only cover what's most important for our specific task (this is the case study aspect) but we'll also cover related methods (this is the guide aspect) which may prove to be useful in other situations. For example, when we're serving our application, we'll expose our latest model as an API endpoint. However, there are several other popular ways to serving models and we'll briefly illustrate those and talk about advantages / disadvantages.
- *Open source tools*: We will be using only open source tools for this project, with the exception of Google Cloud Platform for storage and compute (free credit will be plenty). The reason we're constraining to open source tools is because:
We can focus on the fundamentals, everyone can do it and you will have much better understanding when you do use a paid tool at work (if you want to).
Large companies that deploy ML to production have complicated and scaled processes that don’t make sense for the vast majority of companies / individuals.

#### Who is the author?
- I've deployed large scale ML systems at Apple as well as smaller systems with constraints at startups and want to share the common principles I've learned along the way.
- I created Made With ML so that the community can explore, learn and build ML and I learned how to build it into an end-to-end product that's currently used by over 20K monthly active users.
- You can learn more at my [personal website](https://goku.me/) or [LinkedIn](https://www.linkedin.com/in/goku/).

#### Why is this free?
This is especially targeted for people who don't have as much opportunity around the world. I firmly believe that creativity and intelligence are randomly distributed but opportunity is siloed. I want to enable more people to create and contribute to innovation.

<hr>

<a href="{% link _courses/index.md %}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-left mr-1"></i>Return to courses</a>