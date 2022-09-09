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
            <!-- DONT FORGET TO CHANGE INSIDE dovs/overrides/newsletter.html -->
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
            <p style="font-size: 0.89rem;">Join <b>40K+ developers</b> in learning how to responsibly deliver value with ML.</p>
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

<!-- Accolades -->
<!-- <div class="ai-center-all" style="margin-top: 2rem;">
    <div class="row">
        <div class="col-md-4" data-aos="fade-up" data-aos-easing="ease-out-cubic" data-aos-duration="1000" data-aos-delay="250">
            <div class="px-md-3 py-md-3 px-5 py-3 ai-header-card">🏆 &nbsp;Among the <a href="https://github.com/GokuMohandas/Made-With-ML" target="_blank">top ML repositories</a> on GitHub.</div>
        </div>
        <div class="col-md-4 mt-4 mt-md-0" data-aos="fade-up" data-aos-easing="ease-out-cubic" data-aos-duration="1000" data-aos-delay="750">
            <div class="px-md-3 py-md-3 px-5 py-3 ai-header-card">❤️ &nbsp;<a href="https://madewithml.com/" target="_blank">40K+ community members</a> and growing.</div>
        </div>
        <div class="col-md-4 mt-4 mt-md-0" data-aos="fade-up" data-aos-easing="ease-out-cubic" data-aos-duration="1000" data-aos-delay="1250">
            <div class="px-md-3 py-md-3 px-5 py-3 ai-header-card">🛠️  &nbsp;<a href="https://youtu.be/CjU_6OaYKpw?t=1009" target="_blank">Highly recommended</a> industry resource.</div>
        </div>
    </div>
</div> -->

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

## Foundations
> Learn the foundations of machine learning through intuitive explanations, clean code and visualizations. &rarr; :fontawesome-brands-github:{ .github } [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML){:target="_blank"}

{% include "styles/foundations.md" %}

<hr>

<h2 id="mlops">MLOps course</h2>
> Learn how to combine machine learning with software engineering to build production-grade applications. &rarr; :fontawesome-brands-github:{ .github } [GokuMohandas/mlops-course](https://github.com/GokuMohandas/mlops-course){:target="_blank"}

{% include "styles/mlops.md" %}

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

## Features

<div class="row ai-features">
    <div class="col-md-6 ai-feature" data-aos="fade-right">
        <div class="ai-feature-header">
        <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M12 2.5c-3.81 0-6.5 2.743-6.5 6.119 0 1.536.632 2.572 1.425 3.56.172.215.347.422.527.635l.096.112c.21.25.427.508.63.774.404.531.783 1.128.995 1.834a.75.75 0 01-1.436.432c-.138-.46-.397-.89-.753-1.357a18.354 18.354 0 00-.582-.714l-.092-.11c-.18-.212-.37-.436-.555-.667C4.87 12.016 4 10.651 4 8.618 4 4.363 7.415 1 12 1s8 3.362 8 7.619c0 2.032-.87 3.397-1.755 4.5-.185.23-.375.454-.555.667l-.092.109c-.21.248-.405.481-.582.714-.356.467-.615.898-.753 1.357a.75.75 0 01-1.437-.432c.213-.706.592-1.303.997-1.834.202-.266.419-.524.63-.774l.095-.112c.18-.213.355-.42.527-.634.793-.99 1.425-2.025 1.425-3.561C18.5 5.243 15.81 2.5 12 2.5zM9.5 21.75a.75.75 0 01.75-.75h3.5a.75.75 0 010 1.5h-3.5a.75.75 0 01-.75-.75zM8.75 18a.75.75 0 000 1.5h6.5a.75.75 0 000-1.5h-6.5z"></path></svg></span>Intuition-first
        </div>
        <div class="ai-feature-text">
        We never jump straight to code, instead we develop an intuition for the concepts first.
        </div>
    </div>
    <div class="col-md-6 ai-feature" data-aos="fade-left">
        <div class="ai-feature-header">
        <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M8 8.807V3.5h-.563a.75.75 0 010-1.5h9.125a.75.75 0 010 1.5H16v5.307l5.125 9.301c.964 1.75-.302 3.892-2.299 3.892H5.174c-1.998 0-3.263-2.142-2.3-3.892L8 8.807zM14.5 3.5h-5V9a.75.75 0 01-.093.362L7.127 13.5h9.746l-2.28-4.138A.75.75 0 0114.5 9V3.5zM4.189 18.832L6.3 15h11.4l2.111 3.832a1.125 1.125 0 01-.985 1.668H5.174a1.125 1.125 0 01-.985-1.668z"></path></svg></span>Hands-on
        </div>
        <div class="ai-feature-text">
        Instead of just discussing MLOps concepts, we code everything (project-based).
        </div>
    </div>
    <div class="col-md-6 ai-feature" data-aos="fade-right">
        <div class="ai-feature-header">
        <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M5.75 21a1.75 1.75 0 110-3.5 1.75 1.75 0 010 3.5zM2.5 19.25a3.25 3.25 0 106.5 0 3.25 3.25 0 00-6.5 0zM5.75 6.5a1.75 1.75 0 110-3.5 1.75 1.75 0 010 3.5zM2.5 4.75a3.25 3.25 0 106.5 0 3.25 3.25 0 00-6.5 0zM18.25 6.5a1.75 1.75 0 110-3.5 1.75 1.75 0 010 3.5zM15 4.75a3.25 3.25 0 106.5 0 3.25 3.25 0 00-6.5 0z"></path><path fill-rule="evenodd" d="M5.75 16.75A.75.75 0 006.5 16V8A.75.75 0 005 8v8c0 .414.336.75.75.75z"></path><path fill-rule="evenodd" d="M17.5 8.75v-1H19v1a3.75 3.75 0 01-3.75 3.75h-7a1.75 1.75 0 00-1.75 1.75H5A3.25 3.25 0 018.25 11h7a2.25 2.25 0 002.25-2.25z"></path></svg></span>Engineering
        </div>
        <div class="ai-feature-text">
        It's not just about ML, it's also about software engineering best practices.
        </div>
    </div>
    <div class="col-md-6 ai-feature" data-aos="fade-left">
        <div class="ai-feature-header">
        <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M20.322.75a10.75 10.75 0 00-7.373 2.926l-1.304 1.23A23.743 23.743 0 0010.103 6.5H5.066a1.75 1.75 0 00-1.5.85l-2.71 4.514a.75.75 0 00.49 1.12l4.571.963c.039.049.082.096.129.14L8.04 15.96l1.872 1.994c.044.047.091.09.14.129l.963 4.572a.75.75 0 001.12.488l4.514-2.709a1.75 1.75 0 00.85-1.5v-5.038a23.741 23.741 0 001.596-1.542l1.228-1.304a10.75 10.75 0 002.925-7.374V2.499A1.75 1.75 0 0021.498.75h-1.177zM16 15.112c-.333.248-.672.487-1.018.718l-3.393 2.262.678 3.223 3.612-2.167a.25.25 0 00.121-.214v-3.822zm-10.092-2.7L8.17 9.017c.23-.346.47-.685.717-1.017H5.066a.25.25 0 00-.214.121l-2.167 3.612 3.223.679zm8.07-7.644a9.25 9.25 0 016.344-2.518h1.177a.25.25 0 01.25.25v1.176a9.25 9.25 0 01-2.517 6.346l-1.228 1.303a22.248 22.248 0 01-3.854 3.257l-3.288 2.192-1.743-1.858a.764.764 0 00-.034-.034l-1.859-1.744 2.193-3.29a22.248 22.248 0 013.255-3.851l1.304-1.23zM17.5 8a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zm-11 13c.9-.9.9-2.6 0-3.5-.9-.9-2.6-.9-3.5 0-1.209 1.209-1.445 3.901-1.49 4.743a.232.232 0 00.247.247c.842-.045 3.534-.281 4.743-1.49z"></path></svg></span>Comprehensive
        </div>
        <div class="ai-feature-text">
        Easily extends to all algorithms, data (text, image, tabular), tools, cloud providers, etc.
        </div>
    </div>
</div>

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

<!-- Audience -->
<h2 class="ai-center-all mt-0 mb-2 md-typeset">Who is this content for?</h2>
<p class="ai-center-all" style="font-size: 0.9rem;">Machine learning is not a separate industry, instead, it's a powerful way of thinking about data that's not reserved for any one type of person.</p>

<div class="row ai-features">
    <div class="col-md-6 ai-feature" data-aos="fade-right">
        <div class="ai-feature-header">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path d="M10.3 8.24a.75.75 0 01-.04 1.06L7.352 12l2.908 2.7a.75.75 0 11-1.02 1.1l-3.5-3.25a.75.75 0 010-1.1l3.5-3.25a.75.75 0 011.06.04zm3.44 1.06a.75.75 0 111.02-1.1l3.5 3.25a.75.75 0 010 1.1l-3.5 3.25a.75.75 0 11-1.02-1.1l2.908-2.7-2.908-2.7z"></path><path fill-rule="evenodd" d="M2 3.75C2 2.784 2.784 2 3.75 2h16.5c.966 0 1.75.784 1.75 1.75v16.5A1.75 1.75 0 0120.25 22H3.75A1.75 1.75 0 012 20.25V3.75zm1.75-.25a.25.25 0 00-.25.25v16.5c0 .138.112.25.25.25h16.5a.25.25 0 00.25-.25V3.75a.25.25 0 00-.25-.25H3.75z"></path></svg></span>Software engineers
        </div>
        <div class="ai-feature-text">
            looking to learn ML and become even better software engineers. ML is integrated into increasingly more products so it's important to know how ML systems operate.
        </div>
    </div>
    <div class="col-md-6 ai-feature" data-aos="fade-left">
        <div class="ai-feature-header">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill-rule="evenodd" d="M8.114 2.094a.75.75 0 01.386.656V9h1.252a.75.75 0 110 1.5H5.75a.75.75 0 010-1.5H7V4.103l-.853.533a.75.75 0 01-.795-1.272l2-1.25a.75.75 0 01.762-.02zm4.889 5.66a.75.75 0 01.75-.75h5.232a.75.75 0 01.53 1.28l-2.776 2.777c.55.097 1.057.253 1.492.483.905.477 1.504 1.284 1.504 2.418 0 .966-.471 1.75-1.172 2.27-.687.511-1.587.77-2.521.77-1.367 0-2.274-.528-2.667-.756a.75.75 0 01.755-1.297c.331.193.953.553 1.912.553.673 0 1.243-.188 1.627-.473.37-.275.566-.635.566-1.067 0-.5-.219-.836-.703-1.091-.538-.284-1.375-.443-2.471-.443a.75.75 0 01-.53-1.28l2.643-2.644h-3.421a.75.75 0 01-.75-.75zM7.88 15.215a1.4 1.4 0 00-1.446.83.75.75 0 01-1.37-.61 2.9 2.9 0 012.986-1.71 2.565 2.565 0 011.557.743c.434.446.685 1.058.685 1.778 0 1.641-1.254 2.437-2.12 2.986-.538.341-1.18.694-1.495 1.273H9.75a.75.75 0 010 1.5h-4a.75.75 0 01-.75-.75c0-1.799 1.337-2.63 2.243-3.21 1.032-.659 1.55-1.031 1.55-1.8 0-.355-.116-.584-.26-.732a1.068 1.068 0 00-.652-.298z"></path></svg></span>Data scientists
        </div>
        <div class="ai-feature-text">
            who want to want to go way beyond developing models in a notebook to wrapping them around robust workflows that enable ML systems to improve over time.
        </div>
    </div>
    <div class="col-md-6 ai-feature" data-aos="fade-right">
        <div class="ai-feature-header">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill-rule="evenodd" d="M12.292 2.06a.75.75 0 00-.584 0L.458 6.81a.75.75 0 000 1.38L4.25 9.793v3.803a2.901 2.901 0 00-1.327.757c-.579.58-.923 1.41-.923 2.43v4.5c0 .248.128.486.335.624.06.04.117.073.22.124.124.062.297.138.52.213.448.149 1.09.288 1.925.288s1.477-.14 1.925-.288c.223-.075.396-.15.52-.213a2.11 2.11 0 00.21-.117A.762.762 0 008 21.28v-4.5c0-1.018-.344-1.85-.923-2.428a2.9 2.9 0 00-1.327-.758v-3.17l5.958 2.516a.75.75 0 00.584 0l5.208-2.2v4.003a2.552 2.552 0 01-.079.085 4.057 4.057 0 01-.849.65c-.826.488-2.255 1.021-4.572 1.021-.612 0-1.162-.037-1.654-.1a.75.75 0 00-.192 1.487c.56.072 1.173.113 1.846.113 2.558 0 4.254-.592 5.334-1.23.538-.316.914-.64 1.163-.896a2.84 2.84 0 00.392-.482h.001A.75.75 0 0019 15v-4.892l4.542-1.917a.75.75 0 000-1.382l-11.25-4.75zM5 15c-.377 0-.745.141-1.017.413-.265.265-.483.7-.483 1.368v4.022c.299.105.797.228 1.5.228s1.201-.123 1.5-.228V16.78c0-.669-.218-1.103-.483-1.368A1.431 1.431 0 005 15zm7-3.564L2.678 7.5 12 3.564 21.322 7.5 12 11.436z"></path></svg></span>College graduates
        </div>
        <div class="ai-feature-text">
            looking to learn the practical skills they'll need for the industry and bridge gap between their university curriculum and what industry expects them to know about ML systems.
        </div>
    </div>
    <div class="col-md-6 ai-feature" data-aos="fade-left">
        <div class="ai-feature-header">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill-rule="evenodd" d="M12 2.5c-3.81 0-6.5 2.743-6.5 6.119 0 1.536.632 2.572 1.425 3.56.172.215.347.422.527.635l.096.112c.21.25.427.508.63.774.404.531.783 1.128.995 1.834a.75.75 0 01-1.436.432c-.138-.46-.397-.89-.753-1.357a18.354 18.354 0 00-.582-.714l-.092-.11c-.18-.212-.37-.436-.555-.667C4.87 12.016 4 10.651 4 8.618 4 4.363 7.415 1 12 1s8 3.362 8 7.619c0 2.032-.87 3.397-1.755 4.5-.185.23-.375.454-.555.667l-.092.109c-.21.248-.405.481-.582.714-.356.467-.615.898-.753 1.357a.75.75 0 01-1.437-.432c.213-.706.592-1.303.997-1.834.202-.266.419-.524.63-.774l.095-.112c.18-.213.355-.42.527-.634.793-.99 1.425-2.025 1.425-3.561C18.5 5.243 15.81 2.5 12 2.5zM9.5 21.75a.75.75 0 01.75-.75h3.5a.75.75 0 010 1.5h-3.5a.75.75 0 01-.75-.75zM8.75 18a.75.75 0 000 1.5h6.5a.75.75 0 000-1.5h-6.5z"></path></svg></span>Product managers
        </div>
        <div class="ai-feature-text">
            who want to develop a technical foundation in MLOps so they can effectively communicate with their technical team and help develop and iterate on applications.
        </div>
    </div>
</div>

<!-- Instructor -->
{% include "styles/instructor.md" %}

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

<!-- Wall of love -->
{% include "styles/testimonials.md" %}

<!-- Citation -->
{% include "styles/cite.md" %}