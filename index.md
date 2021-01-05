---
layout: default
title: "Made With ML: Applied ML · MLOps · Production"
permalink: /

latest:
  -
    title: Preprocessing
    description: Preparing and transforming our data for modeling.
    image: /static/images/courses/applied-ml/preprocessing.png
    url: /courses/applied-ml/preprocessing/
  -
    title: Splitting
    description: Appropriately splitting our dataset (multi-label) for training, validation and testing.
    image: /static/images/courses/applied-ml/splitting.png
    url: /courses/applied-ml/splitting/
  -
    title: Exploratory Data Analysis (EDA)
    description: Exploring our dataset for insights with intention.
    image: /static/images/courses/applied-ml/eda.png
    url: /courses/applied-ml/exploratory-data-analysis/

---

<h1 class="page-title">Learn How to Apply ML</h1>
<hr class="mt-0">
<span class="post-date">Join <b>20K+</b> developers in learning how to responsibly
<a href="{% link _pages/about.md %}">deliver value</a> with applied ML.

- New content delivered monthly to your inbox 👉
<a class="btn btn-sm btn-primary ai-btn-purple-gradient ml-1" href="{% link _pages/subscribe.md %}"><i class="fas fa-envelope mr-2"></i>Subscribe</a>
- Follow on <a href="https://twitter.com/GokuMohandas" target="_blank"><i class="fab fa-twitter ai-color-info"></i></a> and <a href="https://www.linkedin.com/in/goku" target="_blank"><i class="fab fa-linkedin ai-color-primary"></i></a> for tips and conversations.

<hr>

{% assign ml_foundations_course_page = site.courses | where:"title", "ML Foundations · Made With ML" | first %}
{{ ml_foundations_course_page.content }}

<hr>

{% assign applied_ml_course_page = site.courses | where:"title", "Applied ML · Made With ML" | first %}
{{ applied_ml_course_page.content }}

<hr>

{% assign faq_page = site.pages | where:"title", "FAQ · Made With ML" | first %}
{{ faq_page.content }}

<!-- <div class="row">
  <div class="col-md-6">
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🔥 Putting ML in Production! We&#39;re going to publicly develop <a href="https://twitter.com/madewithml?ref_src=twsrc%5Etfw">@madewithml</a>&#39;s first ML service. Here is the broad curriculum: <br><br>- 📦 Product<br>- 🔢 Data<br>- 🤖 Modeling<br>- 📝 Scripting<br>- 🛠 API<br>- 🚀 Production<br><br>More details (lessons, task, etc.) here: <a href="https://t.co/xmMm9XGK9j">https://t.co/xmMm9XGK9j</a><br><br>Thread 👇 <a href="https://t.co/T0uLPb2QbR">pic.twitter.com/T0uLPb2QbR</a></p>&mdash; Goku Mohandas (@GokuMohandas) <a href="https://twitter.com/GokuMohandas/status/1315990996849627136?ref_src=twsrc%5Etfw">October 13, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
  </div>
</div> -->

<!-- <h2 class="page-title mb-4">Latest</h2>
<div class="card-deck">
  {% for item in page.latest %}
    <div class="card ai-card">
      <a href="{{ item.url | absolute_url }}">
        <img class="card-img-top" src="{{ item.image }}" alt="Card image cap">
      </a>
      <div class="card-body">
        <a href="{{ item.url | absolute_url }}">
          <h5 class="card-title mb-2" style="font-size: 0.95rem;">{{ item.title }}</h5>
        </a>
        <p class="card-text" style="font-size: 0.85rem !important;">{{ item.description }}</p>
      </div>
    </div>
  {% endfor %}
</div>

<hr>

{% assign courses_page = site.courses | where:"title", "Courses" | first %}
<h2 class="page-title mb-4">{{ courses_page.title }}</h2>
{{ courses_page.content }} -->

