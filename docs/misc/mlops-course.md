---
title: Home
template: main.html
keywords: mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
---

{% include "styles/page.md" %}

<div class="modal fade" id="newsletterForm" tabindex="-1" role="dialog" aria-labelledby="newsletterFormLabel" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered" role="document">
        <div class="modal-content">
            <div class="modal-header" style="padding: 0.5rem 1rem 0.5rem 1rem;">
                <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
            <!-- DON'T FORGET TO CHANGE INSIDE docs/overrides/newsletter.html -->
            <iframe width="540" height="600" src="https://c8efd03b.sibforms.com/serve/MUIEAMs1dZBzyue8b3i3Gw1PHVEmw4JOmt9cLywb0Z10_6R4KyAdiVxZRc2B0Eq19YaA37r1Tjmj4ESTiMsdWAxUKcyI5ctqqGcBFCdoskkHDydRzYCllDt7UNUQJUVsT9JDJ8a48y54PNbKvWB2mzaWtQlCqztp2aG6h9QC4-Jn2sNlTahB7_yIluIBjKjinOjsVyQERl5gwTI4" frameborder="0" scrolling="auto" allowfullscreen style="display: block;margin-left: auto;margin-right: auto;max-width: 100%;"></iframe>
        </div>
    </div>
</div>

<!-- Hero -->
<div class="row flex-column-reverse flex-md-row">
    <div class="col-md-7" data-aos="fade-right">
        <div class="ai-hero-text">
            <h1 style="margin-bottom: 0rem; color: #000; font-weight: 500;">Made With ML</h1>
            <p style="margin-top: 0rem; margin-bottom: 0rem !important; color: #807e7e;">Applied ML · MLOps · Production</p>
            <p style="font-size: 0.89rem;">Join <b>30K+ developers</b> in learning how to responsibly develop, deploy & maintain ML.</p>
            <input class="revue-form-field" placeholder="Your personal email address..." type="email" name="member[email]" id="member_email" style="width: 80%; border: 1px solid #b3b3b3; border-radius: 3px;">
            <button class="md-button md-button--purple-gradient mr-2 mb-2 mb-md-0 mt-md-2 mt-2" style="cursor: pointer !important;" data-toggle="modal" data-target="#newsletterForm">
                <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M1.75 3A1.75 1.75 0 000 4.75v14c0 .966.784 1.75 1.75 1.75h20.5A1.75 1.75 0 0024 18.75v-14A1.75 1.75 0 0022.25 3H1.75zM1.5 4.75a.25.25 0 01.25-.25h20.5a.25.25 0 01.25.25v.852l-10.36 7a.25.25 0 01-.28 0l-10.36-7V4.75zm0 2.662V18.75c0 .138.112.25.25.25h20.5a.25.25 0 00.25-.25V7.412l-9.52 6.433c-.592.4-1.368.4-1.96 0L1.5 7.412z"></path></svg></span> Subscribe</button>
                <a href="#foundations"><span class="md-button md-button--grey-secondary mr-2 mb-2 mb-md-0 mt-md-2 px-3 py-1">View lessons</span></a>
        </div>
    </div>
    <div class="col-md-5 ai-center-all" data-aos="fade-left">
        <div class="mb-md-0 mb-4">
            <img src="/static/images/logos.png" style="width: 12rem; border-radius: 10px;" alt="machine learning logos">
        </div>
    </div>
</div>

{% include "templates/accolades.md" %}

<hr style="margin-top: 2rem; margin-bottom: 0rem;">

<!-- Course header -->>
<section id="interactive-course" data-aos="zoom-in" data-aos-delay="1500" class="p-4">
    <h2 class="ai-center-all mt-1 mb-0">Interactive MLOps Course</h2>
    <div class="ai-center-all">
        <p class="mt-3" style="font-size: 0.83rem;">While all the lessons below are <b>100% free</b>, it's hard to learn everything on your own. That's why we're offering an <i>interactive course</i> with the <i>structure</i> and <i>community</i> to actually complete and master these lessons.</p>
    </div>
    <div class="mb-4 px-4 py-3" style="background-color: #f5f9fd; text-align: center;">
        <a href="#features">&nbsp;⚙️&nbsp; Features &nbsp;</a>| <a href="#alumni-reviews">&nbsp;&nbsp;📝&nbsp; Reviews &nbsp;</a>| <a href="#instructor">&nbsp;&nbsp;🎓&nbsp; Instructor &nbsp;</a>| <a href="#schedule">&nbsp;&nbsp;📆&nbsp; Schedule &nbsp;</a>| <a href="#pricing">&nbsp;&nbsp;💸&nbsp; Pricing &nbsp;</a>|<a href="#faq">&nbsp;❓&nbsp;FAQ</a>
    </div>
    <div class="ai-center-all">
        <a href="#pricing" class="md-button md-button--green-gradient mb-2 mb-md-0 mt-md-0 mt-1" style="cursor: pointer !important;"><span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M20.322.75a10.75 10.75 0 00-7.373 2.926l-1.304 1.23A23.743 23.743 0 0010.103 6.5H5.066a1.75 1.75 0 00-1.5.85l-2.71 4.514a.75.75 0 00.49 1.12l4.571.963c.039.049.082.096.129.14L8.04 15.96l1.872 1.994c.044.047.091.09.14.129l.963 4.572a.75.75 0 001.12.488l4.514-2.709a1.75 1.75 0 00.85-1.5v-5.038a23.741 23.741 0 001.596-1.542l1.228-1.304a10.75 10.75 0 002.925-7.374V2.499A1.75 1.75 0 0021.498.75h-1.177zM16 15.112c-.333.248-.672.487-1.018.718l-3.393 2.262.678 3.223 3.612-2.167a.25.25 0 00.121-.214v-3.822zm-10.092-2.7L8.17 9.017c.23-.346.47-.685.717-1.017H5.066a.25.25 0 00-.214.121l-2.167 3.612 3.223.679zm8.07-7.644a9.25 9.25 0 016.344-2.518h1.177a.25.25 0 01.25.25v1.176a9.25 9.25 0 01-2.517 6.346l-1.228 1.303a22.248 22.248 0 01-3.854 3.257l-3.288 2.192-1.743-1.858a.764.764 0 00-.034-.034l-1.859-1.744 2.193-3.29a22.248 22.248 0 013.255-3.851l1.304-1.23zM17.5 8a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zm-11 13c.9-.9.9-2.6 0-3.5-.9-.9-2.6-.9-3.5 0-1.209 1.209-1.445 3.901-1.49 4.743a.232.232 0 00.247.247c.842-.045 3.534-.281 4.743-1.49z"></path></svg></span> Apply</a>
    </div>
    <div class="ai-center-all mt-3">
        <small><b>Deadline:</b> Sept 30th, 2022</small>
    </div>
    <div class="ai-center-all mt-0">
        <small><b>Start date:</b> Oct 1st, 2022</small>
    </div>
    <div class="ai-center-all mt-1">
        <small>(less than 20 seats remaining!)</small>
    </div>
</section>

<hr style="margin-top: 1rem; margin-bottom: 2rem;">

## Foundations
> Learn the foundations of machine learning through intuitive explanations, clean code and visualizations. &rarr; :fontawesome-brands-github:{ .github } [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML){:target="_blank"}

{% include "templates/foundations.md" %}

<hr>

<h2 id="mlops">MLOps course</h2>
> Learn how to combine machine learning with software engineering to develop, deploy & maintain production ML applications. &rarr; :fontawesome-brands-github:{ .github } [GokuMohandas/mlops-course](https://github.com/GokuMohandas/mlops-course){:target="_blank"}

{% include "templates/mlops.md" %}

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

{% include "templates/features.md" %}

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

{% include "templates/alumni-reviews.md" %}

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

{% include "templates/instructor.md" %}

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

{% include "templates/schedule.md" %}

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

{% include "templates/pricing.md" %}

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

{% include "templates/faq.md" %}

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

{% include "templates/wall_of_love.md" %}

{% include "templates/cite.md" %}