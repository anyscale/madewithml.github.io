---
layout: page
title: Solution
description: Designing a solution with constraints.
image: /assets/images/courses/applied-ml-in-production/evaluation.png
tags: product product-management problem-solving

course-url: /courses/applied-ml-in-production/
next-lesson-url: /courses/applied-ml-in-production/evaluation/
---

<!-- Header -->
<div class="row">
  <div class="col-md-8 col-6 mr-auto">
    <h1 class="page-title">{{ page.title }}</h1>
  </div>
  <div class="col-md-4 col-6">
    <div class="btn-group float-right mb-0" role="group">
      <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i
          class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
    </div>
  </div>
</div>
<hr class="mt-0">

<h3><u>Intuition</u></h3>
Once we've identified our main objective, we can hypothesize solutions using a three-step process.

<table>
    <tr>
        <td style="text-align: center;"><b>Visualize</b></td>
        <td class="p-4">
            Visualize an ideal solution to our problem <b>without</b> factoring in constraints.
            It may seem like a waste of time to think freely without constraints, but it's a chance to think creatively.
            <ul class="my-1">
                <li>most creative solutions start from a blank slate</li>
                <li>void of the bias from previous approaches</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="text-align: center;"><b>Understand</b></td>
        <td class="p-4">
            Understand how the problem is currently being solved (if at all) and the <b>how</b> and <b>why</b> things are currently done the way they are.
            <ul class="my-1">
                <li>prevents us from reinventing the wheel</li>
                <li>gives insight into processes and signals</li>
                <li>opportunity to question everything</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="text-align: center;"><b>Design</b></td>
        <td class="p-4">
            Design from our ideal solution while factoring in <b>constraints</b>.
            <ul class="my-1">
                <li>Automate or augment?</li>
                    <ul class="mb-0">
                        <li>be wary of completely removing the user</li>
                        <li>transition from augment to automate as trust grows</li>
                    </ul>
                <li>UX contraints</li>
                    <ul class="mb-0">
                        <li>privacy, personalization, property</li>
                        <li>dictate the components of our solution</li>
                    </ul>
                <li>Tech constraints</li>
                    <ul class="mb-0">
                        <li>data, time, performance, cost, interpretability, latency</li>
                        <li>dictate the complexity of our solutions</li>
                    </ul>
            </ul>
        </td>
    </tr>
</table>

> The main goal here is to think like a problem solver, as opposed to a naive model fitter.
<table class="mb-0">
  <tr style="text-align: center;">
    <td><b>❌ Model fitter</b></td>
    <td><b>✅ Problem solver</b></td>
  </tr>
  <tr>
    <td><i>naively maps</i> a set of inputs to outputs</td>
    <td>knows which set of inputs and outputs are <i>worth mapping</i></td>
  </tr>
  <tr>
    <td>obsesses on <i>methods</i> (models, SOTA, single metric, etc.)</td>
    <td>focuses on <i>product</i> (objective, constraints, evaluation, etc.)</td>
  </tr>
  <tr>
    <td>fitting methods are <i>ephemeral</i></td>
    <td>foundational mental models are <i>enduring</i></td>
  </tr>
</table>

<h3><u>Application</u></h3>

Our main objective is to allow users to discover the precise resource.

1. **Visualize** The ideal solution would be to ensure that all projects have the proper metadata (tags) so users can discover them. "Ensure proper metadata" is easier said than done so our solution will involve satisfying this requirement through creative ways.

2. **Understand** So far users search for projects using tags. It's important to note here that there are other available signals about each project such as the title, description, details, etc. which are not used in the search process. This is good time to ask why we only rely on tags as opposed to the full text available? Tags are added by the project's author and they represent core concepts that the project covers. This is more meaningful than keywords found in the project's details because the presence of a keyword does not necessarily signify that it's core concept. Additionally, many tags are inferred and don't explicitly exist in the metadata such as `natural-language-processing`, etc.

3. **Design** So we would like all projects to have the appropriate tags and we have the necessary information (title, description, etc.) to meet that requirement. We will decide to augment the user with our solution as opposed to automating the process of adding tags. This is so we can ensure that our suggested tags are in fact relevant to the project and this gives us an opportunity to use the author's decision as feedback for our solution.

<figure>
  <img src="/assets/images/courses/applied-ml-in-production/suggested_tags.png" width="550" alt="pivot">
  <figcaption>UX of our hypothetical solution</figcaption>
</figure>

As for the constraints, we will need to maintain low latency (>100ms) when providing our generated tags since authors complete the entire process within a minute. We also want to keep an eye on the number of tags we suggest because suggesting too many will clutter the screen and overwhelm the user. We'll talk more about how to monitor all of these constraints in the next few lessons.

> For the purpose of this course, we're going to develop a solution that involve applied machine learning in production. However, we would also do A/B testing with other approaches such as simply altering the process where users add tags to projects. So far, the tagging process involves adding tags into an input box but what if we could separate the process into sections like `frameworks`, `tasks`, `algorithms`, etc. to guide the user to add relevant tags. This is a simple solution that needs to be tested against other approaches are effectiveness.


<h3><u>Resources</u></h3>
- [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/research/uploads/prod/2019/01/AI-Guidelines-poster_nogradient_final.pdf){:target="_blank"}
- [People + AI Guidebook](https://pair.withgoogle.com/guidebook/){:target="_blank"}

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

