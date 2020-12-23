---
layout: page
title: Applied ML
description: "A hands-on course on MLOps for software engineers, data scientists and product managers."
image: /static/images/applied_ml.png
redirect_from: /courses/applied-ml/

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

A hands-on course on MLOps for software engineers, data scientists and product managers. We will be developing an end-to-end ML feature using open source tools.

> <i class="fab fa-github ai-color-black mr-1"></i>: [GokuMohandas/applied-ml](https://github.com/GokuMohandas/applied-ml){:target="_blank"}

<hr>

<h2><u>Lessons</u></h2>

<div class="row mt-4">
  <div class="col-md-4">
    <b><span class="mr-1">📦</span> Product</b>
    <ul>
      <li><a href="{% link _courses/applied-ml/objective.md %}">Objective</a></li>
      <li><a href="{% link _courses/applied-ml/solution.md %}">Solution</a></li>
      <li><a href="{% link _courses/applied-ml/evaluation.md %}">Evaluation</a></li>
      <li><a href="{% link _courses/applied-ml/iteration.md %}">Iteration</a></li>
    </ul>
    <b><span class="mr-1">🔢</span> Data</b>
    <ul>
      <li><a href="{% link _courses/applied-ml/annotation.md %}">Annotation</a></li>
      <li><a href="{% link _courses/applied-ml/exploratory-data-analysis.md %}">Exploratory data analysis</a></li>
      <li><a href="{% link _courses/applied-ml/splitting.md %}">Splitting</a></li>
      <li><a href="{% link _courses/applied-ml/preprocessing.md %}">Preprocessing</a></li>
    </ul>
    <b><span class="mr-1">📈</span> Modeling</b>
    <ul>
      <li><a href="{% link _courses/applied-ml/baselines.md %}">Baselines</a></li>
      <li>Experiment tracking</li>
      <li>Optimization</li>
    </ul>
  </div>
  <div class="col-md-4">
    <b><span class="mr-1">📝</span> Scripting</b>
    <ul>
      <li>OOPs</li>
      <li>Formatting</li>
      <li>Packaging</li>
      <li>Logging</li>
    </ul>
    <b><span class="mr-1">✅</span> Testing</b>
    <ul>
      <li>Testing <small>(code)</small></li>
      <li>Testing <small>(data)</small></li>
      <li>Testing <small>(model)</small></li>
    </ul>
    <b><span class="mr-1">⏰</span> Version control</b>
    <ul>
      <li>Git</li>
      <li>Precommit</li>
      <li>Makefile</li>
      <li>Versioning</li>
    </ul>
  </div>
  <div class="col-md-4">
    <b><span class="mr-1">🛠</span> API</b>
    <ul>
      <li>RESTful API</li>
      <li>Databases</li>
      <li>Authentication</li>
      <li>Documentation</li>
    </ul>
    <b><span class="mr-1">🚀</span> Production</b>
    <ul>
      <li>Dashboard</li>
      <li>Docker</li>
      <li>Serving</li>
      <li>CI/CD</li>
      <li>Monitoring <small>(performance, drift)</small></li>
      <li>Active learning</li>
      <li>Feature stores</li>
      <li>Scaling</li>
    </ul>
  </div>
</div>

<span class="ml-1 mr-1"> 📆 </span> new lesson every week!

> <i class="fas fa-info-circle mr-1"></i> If are are planning to use this as a guide for applying ML in production, be aware that it takes **a lot** of effort (initial, maintenance, iteration) compared to deploying traditional software. The use case should demand large scale experimentation where small improvements provide large business impact.


<hr>

<h2><u>FAQ</u></h2>

#### Who is this course for?
- ML developers looking to become end-to-end ML developers.
- Software engineers looking to learn how to responsibly deliver value with applied ML.
- Product managers who want to have a comprehensive understanding of the different stages of ML dev.

#### What is the structure?
Lessons will be released weekly and each one will include:
- `intuition`: high level overview of the concepts that will be covered and how it all fits together.
- `code`: simple code examples to illustrate the concept.
- `application`: applying the concept to our specific task.
- `extensions`: brief look at other tools and techniques that will be useful for difference situations.

#### What are the prerequisites?
You should have some familiarity with Python and [ML foundations](https://github.com/GokuMohandas/madewithml){:target="_blank"}. While we will be experimenting with complex model architectures, you can easily apply the lessons to any class of ML models.

#### What makes this course unique?
- `hands-on`: If you search production ML or MLOps online, you'll find great blog posts and tweets. But in order to really understand these concepts, you need to implement them. Unfortunately, you don’t see a lot of the inner workings of running production ML because of scale, proprietary content & expensive tools. However, Made With ML is free, open and live which makes it a perfect learning opportunity for the community.
- `intuition-first`: We will never jump straight to code. In every lesson, we will develop intuition for the concepts and think about it from a product perspective.
- `software engineering`: This course isn't just about ML. In fact, it's mostly about clean software engineering! We'll cover important concepts like versioning, testing, logging, etc. that really makes this a production-grade product.
- `focused yet holistic`: For every concept, we'll not only cover what's most important for our specific task (this is the case study aspect) but we'll also cover related methods (this is the guide aspect) which may prove to be useful in other situations. For example, when we're serving our application, we'll expose our latest model as an API endpoint. However, there are several other popular ways to serving models and we'll briefly illustrate those and talk about advantages / disadvantages.
- `open-source`: We will be using only open source tools for this project, with the exception of Google Cloud Platform for storage and compute (free credit will be plenty). The reason we're constraining to open source tools is because:
We can focus on the fundamentals, everyone can do it and you will have much better understanding when you do use a paid tool at work (if you want to).
Large companies that deploy ML to production have complicated and scaled processes that don’t make sense for the vast majority of companies / individuals.

#### Who is the author?
- I've deployed large scale ML systems at Apple as well as smaller systems with constraints at startups and want to share the common principles I've learned along the way.
- I created Made With ML so that the community can explore, learn and build ML and I learned how to build it into an end-to-end product that's currently used by over 20K monthly active users.
- Connect with me on <a href="https://twitter.com/GokuMohandas" target="_blank"><i class="fab fa-twitter ai-color-info mr-1"></i>Twitter</a> and <a href="https://www.linkedin.com/in/goku" target="_blank"><i class="fab fa-linkedin ai-color-primary mr-1"></i>LinkedIn</a>

#### Why is this free?
While this is for everyone, this content is especially targeted towards people who don't have as much opportunity to learn around the world. I firmly believe that creativity and intelligence are randomly distributed but opportunity is siloed. I want to enable more people to create and contribute to innovation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🔥 Putting ML in Production! We&#39;re going to publicly develop <a href="https://twitter.com/madewithml?ref_src=twsrc%5Etfw">@madewithml</a>&#39;s first ML service. Here is the broad curriculum: <br><br>- 📦 Product<br>- 🔢 Data<br>- 🤖 Modeling<br>- 📝 Scripting<br>- 🛠 API<br>- 🚀 Production<br><br>More details (lessons, task, etc.) here: <a href="https://t.co/xmMm9XGK9j">https://t.co/xmMm9XGK9j</a><br><br>Thread 👇 <a href="https://t.co/T0uLPb2QbR">pic.twitter.com/T0uLPb2QbR</a></p>&mdash; Goku Mohandas (@GokuMohandas) <a href="https://twitter.com/GokuMohandas/status/1315990996849627136?ref_src=twsrc%5Etfw">October 13, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<hr>

<a href="{% link _courses/index.md %}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-left mr-1"></i>Return to courses</a>