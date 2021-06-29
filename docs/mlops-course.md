---
template: course.html
title: MLOps Course
keywords: mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
---

{% include "styles/page.md" %}

<!-- Hero -->
<div class="row flex-column-reverse flex-md-row">
    <div class="col-md-7">
        <div class="ai-hero-text">
            <h1 style="margin-bottom: 0rem; color: #000; font-weight: 500;">Made With ML</h1>
            <p style="margin-top: 0rem; margin-bottom: 0rem !important; color: #807e7e;">Applied ML · MLOps · Production</p>
            <p style="font-size: 0.89rem;">Join <b>30K+ developers</b> in learning how to responsibly <a href="about">deliver value</a> with ML.</p>
            <div id="revue-embed">
                <form action="https://www.getrevue.co/profile/madewithml/add_subscriber" method="post" id="revue-form" name="revue-form"  target="_blank">
                    <input class="revue-form-field" placeholder="Your email address..." type="email" name="member[email]" id="member_email" style="width: 80%; border: 1px solid #b3b3b3; border-radius: 3px;">
                    <button class="md-button md-button--purple-gradient mr-2 mb-2 mb-md-0 mt-md-2 mt-2" type="submit" style="cursor: pointer !important;">
                        <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M1.75 3A1.75 1.75 0 000 4.75v14c0 .966.784 1.75 1.75 1.75h20.5A1.75 1.75 0 0024 18.75v-14A1.75 1.75 0 0022.25 3H1.75zM1.5 4.75a.25.25 0 01.25-.25h20.5a.25.25 0 01.25.25v.852l-10.36 7a.25.25 0 01-.28 0l-10.36-7V4.75zm0 2.662V18.75c0 .138.112.25.25.25h20.5a.25.25 0 00.25-.25V7.412l-9.52 6.433c-.592.4-1.368.4-1.96 0L1.5 7.412z"></path></svg></span> Subscribe</button>
                        <a href="#basics"><span class="md-button md-button--grey-secondary mr-2 mb-2 mb-md-0 mt-md-2 px-3 py-1">View lessons</span></a>
                </form>
            </div>
        </div>
    </div>
    <div class="col-md-5 ai-center-all">
        <div class="mb-md-0 mb-4">
            <img src="/static/images/logos.png" style="width: 12rem; border-radius: 10px; box-shadow: 0px 0px 15px 0.1px #fff">
        </div>
    </div>
</div>

<div class="ai-center-all" style="margin-top: 1.5rem;">
    <div class="row">
        <div class="col-md-4">
            <div class="px-md-3 py-md-3 px-5 py-3" style="background-color: #f5f9fd; border-radius: 10px;">🏆 &nbsp;Among the <a href="https://github.com/topics/mlops" target="_blank">top MLOps repositories</a> on GitHub.</div>
        </div>
        <div class="col-md-4 mt-2 mt-md-0">
            <div class="px-md-3 py-md-3 px-5 py-3" style="background-color: #f5f9fd; border-radius: 10px;">❤️ &nbsp;<a href="https://newsletter.madewithml.com/" target="_blank">30K+ community members</a> and growing.</div>
        </div>
        <div class="col-md-4 mt-2 mt-md-0">
            <div class="px-md-3 py-md-3 px-5 py-3" style="background-color: #f5f9fd; border-radius: 10px;">🛠️  &nbsp;<a href="https://youtu.be/VSC7WBFMuZo?t=1000" target="_blank">Highly recommended</a> industry resource.</div>
        </div>
    </div>
</div>


<hr style="margin-top: 2rem; margin-bottom: 2rem;">

<!-- Course header -->
<h1 class="ai-center-all mb-0" style="color: rgb(75, 115, 245); font-size: 1rem;">Made With ML</h1>
<h2 class="ai-center-all mt-0 mb-0" style="font-size: 1.5rem;">MLOps Course</h2>

<div class="ai-center-all">
    <p class="mt-3" style="font-size: 0.89rem;">A project-based course on MLOps fundamentals with a focus on intuition and application. All the lessons below are <b>100% free</b> so you can assess the content quality yourself, whereas the 4-week <b>paid course</b> is about the <a href="#features">experience and community</a> to make sure you learn this content well enough to apply it to your own work.</p>
</div>
<div class="ai-center-all">
    <a href="#apply" class="md-button md-button--green-gradient mb-2 mb-md-0 mt-md-0 mt-1" style="cursor: pointer !important;"><span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M20.322.75a10.75 10.75 0 00-7.373 2.926l-1.304 1.23A23.743 23.743 0 0010.103 6.5H5.066a1.75 1.75 0 00-1.5.85l-2.71 4.514a.75.75 0 00.49 1.12l4.571.963c.039.049.082.096.129.14L8.04 15.96l1.872 1.994c.044.047.091.09.14.129l.963 4.572a.75.75 0 001.12.488l4.514-2.709a1.75 1.75 0 00.85-1.5v-5.038a23.741 23.741 0 001.596-1.542l1.228-1.304a10.75 10.75 0 002.925-7.374V2.499A1.75 1.75 0 0021.498.75h-1.177zM16 15.112c-.333.248-.672.487-1.018.718l-3.393 2.262.678 3.223 3.612-2.167a.25.25 0 00.121-.214v-3.822zm-10.092-2.7L8.17 9.017c.23-.346.47-.685.717-1.017H5.066a.25.25 0 00-.214.121l-2.167 3.612 3.223.679zm8.07-7.644a9.25 9.25 0 016.344-2.518h1.177a.25.25 0 01.25.25v1.176a9.25 9.25 0 01-2.517 6.346l-1.228 1.303a22.248 22.248 0 01-3.854 3.257l-3.288 2.192-1.743-1.858a.764.764 0 00-.034-.034l-1.859-1.744 2.193-3.29a22.248 22.248 0 013.255-3.851l1.304-1.23zM17.5 8a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zm-11 13c.9-.9.9-2.6 0-3.5-.9-.9-2.6-.9-3.5 0-1.209 1.209-1.445 3.901-1.49 4.743a.232.232 0 00.247.247c.842-.045 3.534-.281 4.743-1.49z"></path></svg></span> Apply</a>
</div>
<div class="ai-center-all mt-1">
    <small>* Enterprise workshops available</small>
</div>
<div class="ai-center-all">
    <div class="p-3" style="width: 15rem;">
        <div class="ai-center-all">
            <small><b>Application deadline:</b> July 14th, 2021</small>
        </div>
        <div class="ai-center-all">
            <small><b>Course start date:</b> July 17th, 2021</small>
        </div>
    </div>
</div>

<hr style="margin-top: 1rem; margin-bottom: 2rem;">

<!-- Features -->
<div class="row mt-0">
    <div class="col-4 col-md-8">
        <h2 id="features" class="mt-0">Features</h2>
    </div>
    <div class="col-8 col-md-4 text-right" style="font-size: 0.9rem;">
        <small><span style="color: rgb(75, 115, 245);">▓</span> <b>=</b> <span style="color: rgb(75, 115, 245);"><b>paid course features</b></span></small>
    </div>
</div>
<div class="row ai-features">
    <div class="col-md-6 ai-feature">
        <div class="ai-feature-header">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M12 2.5c-3.81 0-6.5 2.743-6.5 6.119 0 1.536.632 2.572 1.425 3.56.172.215.347.422.527.635l.096.112c.21.25.427.508.63.774.404.531.783 1.128.995 1.834a.75.75 0 01-1.436.432c-.138-.46-.397-.89-.753-1.357a18.354 18.354 0 00-.582-.714l-.092-.11c-.18-.212-.37-.436-.555-.667C4.87 12.016 4 10.651 4 8.618 4 4.363 7.415 1 12 1s8 3.362 8 7.619c0 2.032-.87 3.397-1.755 4.5-.185.23-.375.454-.555.667l-.092.109c-.21.248-.405.481-.582.714-.356.467-.615.898-.753 1.357a.75.75 0 01-1.437-.432c.213-.706.592-1.303.997-1.834.202-.266.419-.524.63-.774l.095-.112c.18-.213.355-.42.527-.634.793-.99 1.425-2.025 1.425-3.561C18.5 5.243 15.81 2.5 12 2.5zM9.5 21.75a.75.75 0 01.75-.75h3.5a.75.75 0 010 1.5h-3.5a.75.75 0 01-.75-.75zM8.75 18a.75.75 0 000 1.5h6.5a.75.75 0 000-1.5h-6.5z"></path></svg></span>Intuition-first
        </div>
        <div class="ai-feature-text">
            We'll never jump straight to code, instead we'll develop an intuition for the concepts first.
        </div>
    </div>
    <div class="col-md-6 ai-feature">
        <div class="ai-feature-header">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M8 8.807V3.5h-.563a.75.75 0 010-1.5h9.125a.75.75 0 010 1.5H16v5.307l5.125 9.301c.964 1.75-.302 3.892-2.299 3.892H5.174c-1.998 0-3.263-2.142-2.3-3.892L8 8.807zM14.5 3.5h-5V9a.75.75 0 01-.093.362L7.127 13.5h9.746l-2.28-4.138A.75.75 0 0114.5 9V3.5zM4.189 18.832L6.3 15h11.4l2.111 3.832a1.125 1.125 0 01-.985 1.668H5.174a1.125 1.125 0 01-.985-1.668z"></path></svg></span>Hands-on
        </div>
        <div class="ai-feature-text">
            Instead of just discussing MLOps concepts, we code everything (project-based).
        </div>
    </div>
    <div class="col-md-6 ai-feature">
        <div class="ai-feature-header">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M5.75 21a1.75 1.75 0 110-3.5 1.75 1.75 0 010 3.5zM2.5 19.25a3.25 3.25 0 106.5 0 3.25 3.25 0 00-6.5 0zM5.75 6.5a1.75 1.75 0 110-3.5 1.75 1.75 0 010 3.5zM2.5 4.75a3.25 3.25 0 106.5 0 3.25 3.25 0 00-6.5 0zM18.25 6.5a1.75 1.75 0 110-3.5 1.75 1.75 0 010 3.5zM15 4.75a3.25 3.25 0 106.5 0 3.25 3.25 0 00-6.5 0z"></path><path fill-rule="evenodd" d="M5.75 16.75A.75.75 0 006.5 16V8A.75.75 0 005 8v8c0 .414.336.75.75.75z"></path><path fill-rule="evenodd" d="M17.5 8.75v-1H19v1a3.75 3.75 0 01-3.75 3.75h-7a1.75 1.75 0 00-1.75 1.75H5A3.25 3.25 0 018.25 11h7a2.25 2.25 0 002.25-2.25z"></path></svg></span>Engineering
        </div>
        <div class="ai-feature-text">
            It's not just about ML, it's also about software engineering best practices.
        </div>
    </div>
    <div class="col-md-6 ai-feature">
        <div class="ai-feature-header">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M12 1C5.925 1 1 5.925 1 12s4.925 11 11 11 11-4.925 11-11S18.075 1 12 1zM2.513 11.5h4.745c.1-3.037 1.1-5.49 2.093-7.204.39-.672.78-1.233 1.119-1.673C6.11 3.329 2.746 7 2.513 11.5zm4.77 1.5H2.552a9.505 9.505 0 007.918 8.377 15.698 15.698 0 01-1.119-1.673C8.413 18.085 7.47 15.807 7.283 13zm1.504 0h6.426c-.183 2.48-1.02 4.5-1.862 5.951-.476.82-.95 1.455-1.304 1.88L12 20.89l-.047-.057a13.888 13.888 0 01-1.304-1.88C9.807 17.5 8.969 15.478 8.787 13zm6.454-1.5H8.759c.1-2.708.992-4.904 1.89-6.451.476-.82.95-1.455 1.304-1.88L12 3.11l.047.057c.353.426.828 1.06 1.304 1.88.898 1.548 1.79 3.744 1.89 6.452zm1.476 1.5c-.186 2.807-1.13 5.085-2.068 6.704-.39.672-.78 1.233-1.118 1.673A9.505 9.505 0 0021.447 13h-4.731zm4.77-1.5h-4.745c-.1-3.037-1.1-5.49-2.093-7.204-.39-.672-.78-1.233-1.119-1.673 4.36.706 7.724 4.377 7.957 8.877z"></path></svg></span>Comprehensive
        </div>
        <div class="ai-feature-text">
            Easily extends to all algorithms, data (text, image, tabular), tools, cloud providers, etc.
        </div>
    </div>
    <div class="col-md-6 ai-feature">
        <div class="ai-feature-header" style="color: rgb(75, 115, 245);">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M10.3 8.24a.75.75 0 01-.04 1.06L7.352 12l2.908 2.7a.75.75 0 11-1.02 1.1l-3.5-3.25a.75.75 0 010-1.1l3.5-3.25a.75.75 0 011.06.04zm3.44 1.06a.75.75 0 111.02-1.1l3.5 3.25a.75.75 0 010 1.1l-3.5 3.25a.75.75 0 11-1.02-1.1l2.908-2.7-2.908-2.7z"></path><path fill-rule="evenodd" d="M2 3.75C2 2.784 2.784 2 3.75 2h16.5c.966 0 1.75.784 1.75 1.75v16.5A1.75 1.75 0 0120.25 22H3.75A1.75 1.75 0 012 20.25V3.75zm1.75-.25a.25.25 0 00-.25.25v16.5c0 .138.112.25.25.25h16.5a.25.25 0 00.25-.25V3.75a.25.25 0 00-.25-.25H3.75z"></path></svg></span>Interactive
        </div>
        <div class="ai-feature-text">
            Live lectures, coding sessions & assigments along the way to solidify what we're learning.
        </div>
    </div>
    <div class="col-md-6 ai-feature">
        <div class="ai-feature-header" style="color: rgb(75, 115, 245);">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M3.5 8a5.5 5.5 0 118.596 4.547 9.005 9.005 0 015.9 8.18.75.75 0 01-1.5.045 7.5 7.5 0 00-14.993 0 .75.75 0 01-1.499-.044 9.005 9.005 0 015.9-8.181A5.494 5.494 0 013.5 8zM9 4a4 4 0 100 8 4 4 0 000-8z"></path><path d="M17.29 8c-.148 0-.292.01-.434.03a.75.75 0 11-.212-1.484 4.53 4.53 0 013.38 8.097 6.69 6.69 0 013.956 6.107.75.75 0 01-1.5 0 5.193 5.193 0 00-3.696-4.972l-.534-.16v-1.676l.41-.209A3.03 3.03 0 0017.29 8z"></path></svg></span>Community
        </div>
        <div class="ai-feature-text">
            An alumni community with weekly readings, discussions, brainstorming, etc.
        </div>
    </div>
    <div class="col-md-6 ai-feature">
        <div class="ai-feature-header" style="color: rgb(75, 115, 245);">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M6.736 4C4.657 4 2.5 5.88 2.5 8.514c0 3.107 2.324 5.96 4.861 8.12a29.66 29.66 0 004.566 3.175l.073.041.073-.04c.271-.153.661-.38 1.13-.674.94-.588 2.19-1.441 3.436-2.502 2.537-2.16 4.861-5.013 4.861-8.12C21.5 5.88 19.343 4 17.264 4c-2.106 0-3.801 1.389-4.553 3.643a.75.75 0 01-1.422 0C10.537 5.389 8.841 4 6.736 4zM12 20.703l.343.667a.75.75 0 01-.686 0l.343-.667zM1 8.513C1 5.053 3.829 2.5 6.736 2.5 9.03 2.5 10.881 3.726 12 5.605 13.12 3.726 14.97 2.5 17.264 2.5 20.17 2.5 23 5.052 23 8.514c0 3.818-2.801 7.06-5.389 9.262a31.146 31.146 0 01-5.233 3.576l-.025.013-.007.003-.002.001-.344-.666-.343.667-.003-.002-.007-.003-.025-.013A29.308 29.308 0 0110 20.408a31.147 31.147 0 01-3.611-2.632C3.8 15.573 1 12.332 1 8.514z"></path></svg></span>Accountability
        </div>
        <div class="ai-feature-text">
            Regular check-ins with your fellow peers and the instructor to make sure we're on track.
        </div>
    </div>
    <div class="col-md-6 ai-feature">
        <div class="ai-feature-header" style="color: rgb(75, 115, 245);">
            <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M20.322.75a10.75 10.75 0 00-7.373 2.926l-1.304 1.23A23.743 23.743 0 0010.103 6.5H5.066a1.75 1.75 0 00-1.5.85l-2.71 4.514a.75.75 0 00.49 1.12l4.571.963c.039.049.082.096.129.14L8.04 15.96l1.872 1.994c.044.047.091.09.14.129l.963 4.572a.75.75 0 001.12.488l4.514-2.709a1.75 1.75 0 00.85-1.5v-5.038a23.741 23.741 0 001.596-1.542l1.228-1.304a10.75 10.75 0 002.925-7.374V2.499A1.75 1.75 0 0021.498.75h-1.177zM16 15.112c-.333.248-.672.487-1.018.718l-3.393 2.262.678 3.223 3.612-2.167a.25.25 0 00.121-.214v-3.822zm-10.092-2.7L8.17 9.017c.23-.346.47-.685.717-1.017H5.066a.25.25 0 00-.214.121l-2.167 3.612 3.223.679zm8.07-7.644a9.25 9.25 0 016.344-2.518h1.177a.25.25 0 01.25.25v1.176a9.25 9.25 0 01-2.517 6.346l-1.228 1.303a22.248 22.248 0 01-3.854 3.257l-3.288 2.192-1.743-1.858a.764.764 0 00-.034-.034l-1.859-1.744 2.193-3.29a22.248 22.248 0 013.255-3.851l1.304-1.23zM17.5 8a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zm-11 13c.9-.9.9-2.6 0-3.5-.9-.9-2.6-.9-3.5 0-1.209 1.209-1.445 3.901-1.49 4.743a.232.232 0 00.247.247c.842-.045 3.534-.281 4.743-1.49z"></path></svg></span>Guidance
        </div>
        <div class="ai-feature-text">
            Guidance sessions on how to navigate research papers, libraries and careers.
        </div>
    </div>
</div>

<div class="row mb-0">
    <div class="offset-md-1 col-md-10">
        <div class="admonition warning">
            <p class="admonition-title">This is not a MOOC</p>
            <p>Passive courses on popular online education platforms typically have a <a href="https://www.insidehighered.com/digital-learning/article/2019/01/16/study-offers-data-show-moocs-didnt-achieve-their-goals" target="_blank">2-3% completion rate</a> because they don't offer the cohort-based community experience. Our paid course is interactive and offers accountability to make sure you learn this content well enough to apply it to your own work.</p>
        </div>
    </div>
</div>

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

## Basics
> Learn the fundamentals of ML through intuitive explanations, clean code and visualizations.
[GokuMohandas/MadeWithML](https://github.com/GokuMohandas/MadeWithML){:target="_blank"} &rarr; 🏆 &nbsp;Among the [top ML repos](https://github.com/topics/deep-learning){:target="blank"} on GitHub

<div class="row mt-4 ai-course-list">
<div class="col-md-4">
    <b><span class="mr-1">🔢</span> Foundations</b>
    <ul>
    <li><a href="/courses/basics/notebooks/">Notebooks</a></li>
    <li><a href="/courses/basics/python/">Python</a></li>
    <li><a href="/courses/basics/numpy/">NumPy</a></li>
    <li><a href="/courses/basics/pandas/">Pandas</a></li>
    <li><a href="/courses/basics/pytorch/">PyTorch</a></li>
    </ul>
</div>
<div class="col-md-4">
    <b><span class="mr-1">📈</span> Modeling</b>
    <ul>
    <li><a href="/courses/basics/linear-regression/">Linear Regression</a></li>
    <li><a href="/courses/basics/logistic-regression/">Logistic Regression</a></li>
    <li><a href="/courses/basics/neural-networks/">Neural Networks</a></li>
    <li><a href="/courses/basics/data-quality/">Data Quality</a></li>
    <li><a href="/courses/basics/utilities/">Utilities</a></li>
    </ul>
</div>
<div class="col-md-4">
    <b><span class="mr-1">🤖</span> Deep Learning</b>
    <ul>
    <li><a href="/courses/basics/convolutional-neural-networks/">CNNs</a></li>
    <li><a href="/courses/basics/embeddings/">Embeddings</a></li>
    <li><a href="/courses/basics/recurrent-neural-networks/">RNNs</a></li>
    <li><a href="/courses/mlops/baselines/#transformers-w-contextual-embeddings">Transformers</a></li>
    </ul>
</div>
</div>

<hr>

## MLOps
> Learn how to apply ML to build a production grade product and deliver value.
[GokuMohandas/MLOps](https://github.com/GokuMohandas/MLOps){:target="_blank"} &rarr; 🏆 &nbsp;Among the [top MLOps repos](https://github.com/topics/mlops){:target="blank"} on GitHub

<div class="row mt-4 ai-course-list">
<div class="col-md-4">
    <b><span class="mr-1">📦</span> Product</b>
    <ul>
    <li><a href="/courses/mlops/objective/">Objective</a></li>
    <li><a href="/courses/mlops/solution/">Solution</a></li>
    <li><a href="/courses/mlops/iteration/">Iteration</a></li>
    </ul>
    <b><span class="mr-1">🔢</span> Data</b>
    <ul>
    <li><a href="/courses/mlops/labeling/">Labeling</a></li>
    <li><a href="/courses/mlops/preprocessing/">Preprocessing</a></li>
    <li><a href="/courses/mlops/exploratory-data-analysis/">Exploration</a></li>
    <li><a href="/courses/mlops/splitting/">Splitting</a></li>
    <li><a href="/courses/mlops/augmentation/">Augmentation</a></li>
    </ul>
    <b><span class="mr-1">📈</span> Modeling</b>
    <ul>
    <li><a href="/courses/mlops/baselines/">Baselines</a></li>
    <li><a href="/courses/mlops/evaluation/">Evaluation</a></li>
    <li><a href="/courses/mlops/experiment-tracking/">Experiment tracking</a></li>
    <li><a href="/courses/mlops/optimization/">Optimization</a></li>
    </ul>
</div>
<div class="col-md-4">
    <b><span class="mr-1">📝</span> Scripting</b>
    <ul>
    <li><a href="/courses/mlops/organization/">Organization</a></li>
    <li><a href="/courses/mlops/packaging/">Packaging</a></li>
    <li><a href="/courses/mlops/documentation/">Documentation</a></li>
    <li><a href="/courses/mlops/logging/">Logging</a></li>
    <li><a href="/courses/mlops/styling/">Styling</a></li>
    <li><a href="/courses/mlops/makefile/">Makefile</a></li>
    </ul>
    <b><span class="mr-1">📦</span> Interfaces</b>
    <ul>
    <li><a href="/courses/mlops/cli/">Command-line</a></li>
    <li><a href="/courses/mlops/api/">RESTful API</a></li>
    </ul>
    <b><span class="mr-1">✅</span> Testing</b>
    <ul>
    <li><a href="/courses/mlops/testing/">Code</a></li>
    <li><a href="/courses/mlops/testing/#data">Data</a></li>
    <li><a href="/courses/mlops/testing/#models">Models</a></li>
    </ul>
</div>
<div class="col-md-4">
    <b><span class="mr-1">♻️</span> Reproducibility</b>
    <ul>
    <li><a href="/courses/mlops/git/">Git</a></li>
    <li><a href="/courses/mlops/pre-commit/">Pre-commit</a></li>
    <li><a href="/courses/mlops/versioning/">Versioning</a></li>
    <li><a href="/courses/mlops/docker/">Docker</a></li>
    </ul>
    <b><span class="mr-1">🚀</span> Production</b>
    <ul>
    <li><a href="/courses/mlops/dashboard/">Dashboard</a></li>
    <li><a href="/courses/mlops/cicd/">CI/CD workflows</a></li>
    <li><a href="/courses/mlops/infrastructure/">Infrastructure</a></li>
    <li><a href="/courses/mlops/monitoring/">Monitoring</a></li>
    <li><a href="/courses/mlops/pipelines/">Pipelines</a></li>
    <li><a href="/courses/mlops/feature-store/">Feature store</a></li>
    </ul>
</div>
</div>

<hr style="margin-top: 1rem; margin-bottom: 2rem;">

<h2 id="instructor" class="ai-center-all mt-0 mb-2 md-typeset">Meet your instructor</h2>

<div class="ai-center-all mt-4">
    <img class="ai-header-image" src="/static/images/goku_circle.png">
</div>
<div class="ai-center-all">
    <h3 class="mb-2" style="font-weight: 500 !important;">Hi, I'm Goku Mohandas</h3>
</div>
<div class="ai-center-all mb-4">
    <a href="https://twitter.com/GokuMohandas" target="_blank"><span class="twemoji mr-2" style="font-size: 0.9rem; color: #00b0ff;"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path d="M459.37 151.716c.325 4.548.325 9.097.325 13.645 0 138.72-105.583 298.558-298.558 298.558-59.452 0-114.68-17.219-161.137-47.106 8.447.974 16.568 1.299 25.34 1.299 49.055 0 94.213-16.568 130.274-44.832-46.132-.975-84.792-31.188-98.112-72.772 6.498.974 12.995 1.624 19.818 1.624 9.421 0 18.843-1.3 27.614-3.573-48.081-9.747-84.143-51.98-84.143-102.985v-1.299c13.969 7.797 30.214 12.67 47.431 13.319-28.264-18.843-46.781-51.005-46.781-87.391 0-19.492 5.197-37.36 14.294-52.954 51.655 63.675 129.3 105.258 216.365 109.807-1.624-7.797-2.599-15.918-2.599-24.04 0-57.828 46.782-104.934 104.934-104.934 30.213 0 57.502 12.67 76.67 33.137 23.715-4.548 46.456-13.32 66.599-25.34-7.798 24.366-24.366 44.833-46.132 57.827 21.117-2.273 41.584-8.122 60.426-16.243-14.292 20.791-32.161 39.308-52.628 54.253z"></path></svg></span></a>
    <a href="https://www.linkedin.com/in/goku" target="_blank"><span class="twemoji mr-2" style="font-size: 0.9rem; color: #4051b5;"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><path d="M416 32H31.9C14.3 32 0 46.5 0 64.3v383.4C0 465.5 14.3 480 31.9 480H416c17.6 0 32-14.5 32-32.3V64.3c0-17.8-14.4-32.3-32-32.3zM135.4 416H69V202.2h66.5V416zm-33.2-243c-21.3 0-38.5-17.3-38.5-38.5S80.9 96 102.2 96c21.2 0 38.5 17.3 38.5 38.5 0 21.3-17.2 38.5-38.5 38.5zm282.1 243h-66.4V312c0-24.8-.5-56.7-34.5-56.7-34.6 0-39.9 27-39.9 54.9V416h-66.4V202.2h63.7v29.2h.9c8.9-16.8 30.6-34.5 62.9-34.5 67.2 0 79.7 44.3 79.7 101.9V416z"></path></svg></span></a>
    <a href="https://www.github.com/GokuMohandas" target="_blank"><span class="twemoji" style="font-size: 0.9rem; color: #4c4c4c;"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512"><path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"></path></svg></span></a>
</div>

<p>
    Over the past 7 years, I've worked on ML and product at a large company (<a href="https://www.apple.com/" target="_blank">Apple</a>), a startup in the oncology space (<a href="https://www.ciitizen.com/" target="_blank">Ciitizen</a>) and ran my own startup in the rideshare space (HotSpot). Throughout my journey, I've worked with brilliant developers and product managers and learned how to responsibly develop, deploy and iterate on ML systems across various industries.
</p>
<p>
    I currently work closely with early-stage and mid-sized companies in helping them deliver value with ML while diving into the best and bespoke practices of this rapidly evolving space. I want to share this knowledge with the rest of the world so we can accelerate progress in this space.
</p>
<p>
    ML is <b>not a separate industry</b>, instead, it's a powerful <b>way of thinking</b> about data, so let's make sure we have a solid foundation before we start changing the world. Made With ML is our medium to catalyze this goal and though we're off to great start, we still have a long way to go.
</p>

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

## Schedule

<div class="row">
    <div class="col-md-4 mt-md-0 mt-2">
        <b>
        <div>Saturdays & Sundays</div>
        <small><div>8 am - 11 am PST</div></small>
        </b>
        <div class="mt-2"><span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M8.954 17H2.75A1.75 1.75 0 011 15.25V3.75C1 2.784 1.784 2 2.75 2h18.5c.966 0 1.75.784 1.75 1.75v11.5A1.75 1.75 0 0121.25 17h-6.204c.171 1.375.805 2.652 1.769 3.757A.75.75 0 0116.25 22h-8.5a.75.75 0 01-.565-1.243c.964-1.105 1.598-2.382 1.769-3.757zM21.5 3.75v11.5a.25.25 0 01-.25.25H2.75a.25.25 0 01-.25-.25V3.75a.25.25 0 01.25-.25h18.5a.25.25 0 01.25.25zM13.537 17c.125 1.266.564 2.445 1.223 3.5H9.24c.659-1.055 1.097-2.234 1.223-3.5h3.074z"></path></svg></span> Live classes and coding workshops</div>
    </div>
    <div class="col-md-4 mt-md-0 mt-2">
        <b>
        <div>Tuesdays</div>
        <small><div>5 pm - 6 pm PST</div></small>
        </b>
        <div class="mt-2"><span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M1.75 1A1.75 1.75 0 000 2.75v9.5C0 13.216.784 14 1.75 14H3v1.543a1.457 1.457 0 002.487 1.03L8.061 14h6.189A1.75 1.75 0 0016 12.25v-9.5A1.75 1.75 0 0014.25 1H1.75zM1.5 2.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v9.5a.25.25 0 01-.25.25h-6.5a.75.75 0 00-.53.22L4.5 15.44v-2.19a.75.75 0 00-.75-.75h-2a.25.25 0 01-.25-.25v-9.5z"></path><path d="M22.5 8.75a.25.25 0 00-.25-.25h-3.5a.75.75 0 010-1.5h3.5c.966 0 1.75.784 1.75 1.75v9.5A1.75 1.75 0 0122.25 20H21v1.543a1.457 1.457 0 01-2.487 1.03L15.939 20H10.75A1.75 1.75 0 019 18.25v-1.465a.75.75 0 011.5 0v1.465c0 .138.112.25.25.25h5.5a.75.75 0 01.53.22l2.72 2.72v-2.19a.75.75 0 01.75-.75h2a.25.25 0 00.25-.25v-9.5z"></path></svg></span> Live Q&A office hours</div>
    </div>
    <div class="col-md-4 mt-md-0 mt-2">
        <b>
        <div>Thursdays</div>
        <small><div>5 pm - 6 pm PST</div></small>
        </b>
        <div class="mt-2"><span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M20.322.75a10.75 10.75 0 00-7.373 2.926l-1.304 1.23A23.743 23.743 0 0010.103 6.5H5.066a1.75 1.75 0 00-1.5.85l-2.71 4.514a.75.75 0 00.49 1.12l4.571.963c.039.049.082.096.129.14L8.04 15.96l1.872 1.994c.044.047.091.09.14.129l.963 4.572a.75.75 0 001.12.488l4.514-2.709a1.75 1.75 0 00.85-1.5v-5.038a23.741 23.741 0 001.596-1.542l1.228-1.304a10.75 10.75 0 002.925-7.374V2.499A1.75 1.75 0 0021.498.75h-1.177zM16 15.112c-.333.248-.672.487-1.018.718l-3.393 2.262.678 3.223 3.612-2.167a.25.25 0 00.121-.214v-3.822zm-10.092-2.7L8.17 9.017c.23-.346.47-.685.717-1.017H5.066a.25.25 0 00-.214.121l-2.167 3.612 3.223.679zm8.07-7.644a9.25 9.25 0 016.344-2.518h1.177a.25.25 0 01.25.25v1.176a9.25 9.25 0 01-2.517 6.346l-1.228 1.303a22.248 22.248 0 01-3.854 3.257l-3.288 2.192-1.743-1.858a.764.764 0 00-.034-.034l-1.859-1.744 2.193-3.29a22.248 22.248 0 013.255-3.851l1.304-1.23zM17.5 8a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zm-11 13c.9-.9.9-2.6 0-3.5-.9-.9-2.6-.9-3.5 0-1.209 1.209-1.445 3.901-1.49 4.743a.232.232 0 00.247.247c.842-.045 3.534-.281 4.743-1.49z"></path></svg></span> How to navigate X series</div>
        <small>X = research, libraries, career</small>
    </div>
</div>

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

<p id="apply" class="ai-center-all mt-4" style="font-size: 1rem;">Learn how to responsibly deliver value with ML.</p>
<div class="ai-center-all">
    <div class="p-1" style="width: 15rem;">
        <div class="ai-center-all">
            <small><b>Application deadline:</b> July 14th, 2021</small>
        </div>
        <div class="ai-center-all">
            <small><b>Course start date:</b> July 17th, 2021</small>
        </div>
    </div>
</div>

<div class="row mt-4">
    <div class="col-md-4 mb-md-0 mb-4">
        <div class="card h-100" style="max-width: 18rem;">
            <div class="card-header bg-transparent ai-center-all"><h3 class="mt-0 mb-0">Essential</h3></div>
            <div class="card-body">
                <ul class="task-list">
                    <li class="task-list-item"><label class="task-list-control"><input type="checkbox" checked=""><span class="task-list-indicator"></span></label>  &nbsp;Live sessions and coding workshops</li>
                    <li class="task-list-item"><label class="task-list-control"><input type="checkbox" checked=""><span class="task-list-indicator"></span></label>  &nbsp;Private online community</li>
                    <li class="task-list-item"><label class="task-list-control"><input type="checkbox" checked=""><span class="task-list-indicator"></span></label>  &nbsp;Live Q&A office hours</li>
                    <li class="task-list-item"><label class="task-list-control"><input type="checkbox" checked=""><span class="task-list-indicator"></span></label>  &nbsp;"How to navigate X" series <small>(ML research, Python libraries, career)</small></li>
                </ul>
            </div>
            <div class="card-footer bg-transparent">
                <div class="ai-center-all">
                    <p class="mt-0 mb-1" style="font-size: 0.9rem;">$850 USD</p>
                </div>
                <div class="ai-center-all">
                    <a href="https://forms.gle/6mHLoMzEPx11ZRsK6" target="_blank" class="md-button md-button--purple-gradient mb-2 mb-md-1 mt-md-0 mt-1" style="cursor: pointer !important;">Apply</a>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-4 mb-md-0 mb-4">
        <div class="card h-100" style="max-width: 18rem;">
            <div class="card-header bg-transparent ai-center-all"><h3 class="mt-0 mb-0">Premium</h3></div>
            <div class="card-body">
                <ul class="task-list">
                    <li class="task-list-item"><label class="task-list-control"><input type="checkbox" checked=""><span class="task-list-indicator"></span></label>  &nbsp;<i>&larr; Everything included in the Essential plan</i></li>
                    <li class="task-list-item"><label class="task-list-control"><input type="checkbox" checked=""><span class="task-list-indicator"></span></label>  &nbsp;60 min. 1-on-1 career guidance <div><small>(resume & portfolio)</small></div></li>
                    <li class="task-list-item"><label class="task-list-control"><input type="checkbox" checked=""><span class="task-list-indicator"></span></label>  &nbsp;60 min. 1-on-1 <div>mock interview</div> <div><small>(technical & behavioral)</small></div></li>
                </ul>
            </div>
            <div class="card-footer bg-transparent">
                <div class="ai-center-all">
                    <p class="mt-0 mb-1" style="font-size: 0.9rem;">$1250 USD</p>
                </div>
                <div class="ai-center-all">
                    <a href="https://forms.gle/Cz7omhnToUTpupD98" target="_blank" class="md-button md-button--purple-gradient mb-2 mb-md-1 mt-md-0 mt-1" style="cursor: pointer !important;">Apply</a>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-4 mb-md-0 mb-4">
        <div class="card h-100" style="max-width: 18rem;">
            <div class="card-header bg-transparent ai-center-all"><h3 class="mt-0 mb-0">Enterprise</h3></div>
            <div class="card-body">
                <ul class="task-list">
                    <li class="task-list-item"><label class="task-list-control"><input type="checkbox" checked=""><span class="task-list-indicator"></span></label>  &nbsp;<i>Accelerated</i> live sessions</li>
                    <li class="task-list-item"><label class="task-list-control"><input type="checkbox" checked=""><span class="task-list-indicator"></span></label>  &nbsp;Private online community</li>
                    <li class="task-list-item"><label class="task-list-control"><input type="checkbox" checked=""><span class="task-list-indicator"></span></label>  &nbsp;Live Q&A office hours</li>
                    <li class="task-list-item"><label class="task-list-control"><input type="checkbox" checked=""><span class="task-list-indicator"></span></label>  &nbsp;"How to navigate X" series <small>(ML research, Python libraries, career)</small></li>
                </ul>
            </div>
            <div class="card-footer bg-transparent">
                <div class="ai-center-all">
                    <p class="mt-0 mb-1" style="font-size: 0.8rem;">virtual or in-person</p>
                </div>
                <div class="ai-center-all">
                    <a href="https://forms.gle/8L8ga4fTnmwss9by8" target="_blank" class="md-button md-button--grey-secondary mb-2 mb-md-1 mt-md-0 mt-1" style="cursor: pointer !important;">Contact us</a>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="my-3"></div>

<div class="row mb-0">
    <div class="offset-md-1 col-md-10">
        <div class="admonition info">
            <p class="admonition-title">Reimbursement</p>
            <p>Our alumni have had a lot of success getting 100% of the course reimbursed by their employers since all of this directly falls under career development and, in many cases, highly necessary for their work. After you apply and are accepted for a cohort, check out this <b><a href="https://madewithml.com/reimbursement" target="_blank">email template</a></b> that we've put together that you can send to your manager to get this course reimbursed.</p>
        </div>
    </div>
</div>

<hr style="margin-top: 2rem; margin-bottom: 2rem;">

<!-- FAQ -->
## Frequently Asked Questions (FAQ)

### Who is this course for?

- `#!js Software engineers` looking to learn ML and become even better software engineers.
- `#!js Data scientists` who want to learn how to responsibly deliver value with ML.
- `#!js College graduates` looking to learn the practical skills they'll need for the industry.
- `#!js Product Managers` who want to develop a technical foundation for ML applications.

### What can I get out of this course?

It's extremely difficult to learn most things on our own, let alone a complicated and fast-paced topic like MLOps. This is why I made all of the [content](#basics) completely free so we can see the quality of what we'll learn. The paid course is all about going through this content in just a few weeks with a small, highly motivated cohort. We'll have interactive lectures, assignments, readings, discussions and of course, it's all 100% project based so we actually learn how to apply everything we learn.

> Learning this content will set us up for where ML is headed and because we focus on the foundations, the intuition we'll develop is everlasting and we'll be equipped to adapt to this ever-changing space.

### What are there prerequisites?

During the first week of the course, we're going to cover all of the [Basics](#basics) lessons. We'll be doing this fairly quickly so we can focus on the [MLOps](#mlops) content shortly after. While we will cover the basics of Python and deep learning, it's highly recommended to be familiar with the following:

- Python (variables, lists, dictionaries, functions, classes)
- Scientific computing (NumPy, Pandas)
- PyTorch (nn.module, training/inference loops)
- ML/DL (basics of regression, neural networks, CNN, etc.)

> If you think we should offer a separate, more detailed, course on machine learning fundamentals, [let us know](mailto:hello@madewithml.com){:target="_blank"}!

### Kubernetes (K8s), KubeFlow, AWS, GCP, etc.?

This course covers the fundamentals of MLOps that will easily extend to any type of container-orchestration system, cloud provider, etc. We don’t explicitly use any of these because different situations call for different tools, even within the same company. So it’s important that we have the foundation to adapt to any stack very quickly. We will cover these topics in detail in future iterations of this course as they become more widely adopted.

### Can I do the course while working full-time or going to school?

Absolutely, this course is meant for busy people who are building things, which is why the mandatory sessions are during the weekend.

### Why are the lessons already free?

We're the [top MLOps repository](https://github.com/topics/mlops){:target="_blank"} on GitHub (26K+ stars), a quickly [growing community](https://newsletter.madewithml.com/){:target="_blank"} (30K+) and a [highly recommended](https://youtu.be/VSC7WBFMuZo?t=1000){:target="_blank"} resource used by industry. But we wanted to show everyone exactly what we'll learn in the course because the paid course is all about the [experience](#features).

> The free content is especially targeted towards people who don't have as much opportunity to learn. I firmly believe that creativity and intelligence are randomly distributed but opportunity is siloed. I want to enable more people to create and contribute to innovation.


<div class="row mb-0">
    <div class="offset-md-1 col-md-10">
        <div class="admonition question">
            <p class="admonition-title">More questions?</p> <p>Feel free to send us an <a href="mailto:hello@madewithml.com" target="_blank">email</a> with all your additional questions and we'll get back to you very soon.</p>
        </div>
    </div>
</div>

<hr style="margin-top: 1rem; margin-bottom: 2rem;">

<!-- Citation -->
To cite this content, please use:

```bash linenums="1"
@misc{madewithml,
    title  = "Made With ML",
    author = "Goku Mohandas",
    url    = "https://madewithml.com/"
    year   = "2021",
}
```