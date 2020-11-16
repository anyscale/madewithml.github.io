---
layout: default
title: "Made With ML: Applied ML · MLOps · Production"
permalink: /

latest:
  -
    title: Annotation
    description: Preparing and labeling our data for exploration.
    image: /static/images/courses/applied-ml-in-production/annotation.png
    url: /courses/applied-ml-in-production/annotation/
  -
    title: Iteration
    description: Improving on our solution iteratively over time.
    image: /static/images/courses/applied-ml-in-production/iteration.png
    url: /courses/applied-ml-in-production/iteration/
  -
    title: Evaluation
    description: Determining how well our solution is performing over time.
    image: /static/images/courses/applied-ml-in-production/evaluation.png
    url: /courses/applied-ml-in-production/evaluation/

---

<h1 class="page-title">Welcome to Made With ML</h1>
<hr class="mt-0">
<span class="post-date">Join <b>20K+</b> developers in learning how to
<a href="{% link _pages/about.md %}">deliver (actual) value</a> with machine learning.
<br>⮑ Read about our recent <a href="{% link _pages/pivot.md %}">pivot</a> from our dynamic platform.</span>

### Stay updated
- [Sign up]({{ site.signup_url }}){:target="_blank"} for our newsletter to receive updates on new content.
- [Subscribe](https://www.youtube.com/madewithml?sub_confirmation=1){:target="_blank"} to our new YouTube channel for new lessons.
- Connect with us on [Twitter](https://twitter.com/madewithml){:target="_blank"} and
[LinkedIn](https://www.linkedin.com/company/madewithml){:target="_blank"} for updates on useful resources.

<div class="alert info" role="alert">
  <span style="text-align: left;">
    <i class="fas fa-info-circle mr-1"></i> Connect with <i>Goku Mohandas</i> on
    <a href="https://twitter.com/GokuMohandas" target="_blank">Twitter</a> and
    <a href="https://www.linkedin.com/in/goku" target="_blank">LinkedIn</a> for regular tips and
    interactive conversations on upcoming content.
  </span>
</div>

<hr>

<h2 class="page-title mb-4">Latest</h2>
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
{{ courses_page.content }}

<hr>

{% assign resources_page = site.resources | where:"title", "Resources" | first %}
<h2 class="page-title mb-4">{{ resources_page.title }}</h2>
{{ resources_page.content }}

