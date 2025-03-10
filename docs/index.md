---
title: Home
template: main.html
keywords: mlops
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
            <!-- DON'T FORGET TO CHANGE LINK INSIDE docs/overrides/newsletter.html and mkdocs.yml -->
            <iframe width="540" height="600" src="https://c8efd03b.sibforms.com/serve/MUIFAKa3IQxVRvYHZ_oiARAblHq4WbNhDT72vx1pHJFklbHrp4V813O6mQMUHN5ikC51vZBBw2VqyEgMGgf6NFQg9rC8qgcURZBtzPj5TjOFimUAPyYPTLFrmd6nRKV0OK09SRnZxucZX0xMGR02ADg0GSvd_see2qS0VZnFPJ_JudrivA7uA4fs4BZrNn_3_fMjmF_Bj9ZOD9Ia" frameborder="0" scrolling="auto" allowfullscreen style="display: block;margin-left: auto;margin-right: auto;max-width: 100%;"></iframe>
        </div>
    </div>
</div>

<!-- Hero -->
<div class="row flex-column-reverse flex-md-row">
    <div class="col-md-7" data-aos="fade-right">
        <div class="ai-hero-text">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <h1 style="margin-bottom: 0rem; color: #000; font-weight: 500;">Made With ML</h1>
                <p style="font-weight: 600;">BY</p>
                <a href="https://www.anyscale.com?utm_source=madewithmml&utm_medium=website&utm_campaign=hero" target="_blank"><img src="/static/images/anyscale-black-text.svg" style="width: 8rem;"></a>
            </div>
            <p style="font-size: 0.89rem;">Join <b>40K+ developers</b> in learning how to responsibly deliver value with ML!</p>
            <input class="revue-form-field" placeholder="Work email" type="email" name="member[email]" id="member_email" style="width: 80%; border: 1px solid #b3b3b3; border-radius: 3px;">
            <button class="md-button md-button--purple-gradient mr-2 mb-2 mb-md-0 mt-md-2 mt-2" style="cursor: pointer !important;" data-toggle="modal" data-target="#newsletterForm">
                <span class="twemoji mr-1"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M1.75 3A1.75 1.75 0 000 4.75v14c0 .966.784 1.75 1.75 1.75h20.5A1.75 1.75 0 0024 18.75v-14A1.75 1.75 0 0022.25 3H1.75zM1.5 4.75a.25.25 0 01.25-.25h20.5a.25.25 0 01.25.25v.852l-10.36 7a.25.25 0 01-.28 0l-10.36-7V4.75zm0 2.662V18.75c0 .138.112.25.25.25h20.5a.25.25 0 00.25-.25V7.412l-9.52 6.433c-.592.4-1.368.4-1.96 0L1.5 7.412z"></path></svg></span> Subscribe
            </button>
            <a href="#mlops"><span class="md-button md-button--grey-secondary mr-2 mb-2 mb-md-0 mt-md-2 px-3 py-1">View lessons</span></a>
        </div>
    </div>
    <div class="col-md-5 ai-center-all" data-aos="fade-left">
        <div class="mb-md-0 mb-4">
            <img src="/static/images/logos.png" style="width: 13rem; border-radius: 10px;" alt="machine learning logos">
        </div>
    </div>
</div>

<hr style="margin-top: 2.25rem; margin-bottom: 2.25rem;">

<h2 id="course" class="ai-center-all" style="margin-bottom: 0rem;">ML for Developers</h2>
<p style="margin-top: 0rem; margin-bottom: 0rem !important; color: #807e7e;" class="ai-center-all">Design · Develop · Deploy · Iterate</p>

Learn how to combine machine learning with software engineering to design, develop, deploy and iterate on production ML applications. &rarr; :fontawesome-brands-github:{ .github } [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML){:target="\_blank"}

{% include "templates/mlops.md" %}

<hr style="margin-top: 2.25rem; margin-bottom: 2.25rem;">

<!-- Youtube Video -->
<div class="ai-yt-mobile">
    <iframe id="yt-video-mobile" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" width="620" height="347" type="text/html" src="" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="ai-yt-desktop">
    <iframe id="yt-video-desktop" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" width="620" height="347" type="text/html" src="" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<script>
    var yt_video_id = "AWgkt8H8yVo"
    var yt_video_source = "https://www.youtube.com/embed/" + yt_video_id + "?autoplay=0&fs=1&iv_load_policy=1&showinfo=1&rel=0&cc_load_policy=0&vq=hd720"
    document.getElementById("yt-video-mobile").src = yt_video_source;
    document.getElementById("yt-video-desktop").src = yt_video_source;
</script>

<br>

{% include "templates/values.md" %}

<hr style="margin-top: 2.25rem; margin-bottom: 2.25rem;">

{% include "templates/audience.md" %}

<hr style="margin-top: 2.25rem; margin-bottom: 2.25rem;">

{% include "templates/instructor.md" %}

<hr style="margin-top: 2.25rem; margin-bottom: 2.25rem;">

{% include "templates/wall-of-love.md" %}

{% include "templates/signup.md" %}

<hr style="margin-top: 2.25rem; margin-bottom: 2.25rem;">

{% include "templates/faq.md" %}

{% include "templates/cite.md" %}
