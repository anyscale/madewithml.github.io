<section id="schedule"></section>

<h2 class="ai-center-all mt-0 mb-3 md-typeset">Schedule</h2>

<script>
function get_time(start_str, end_str) {
    var day = new Date(start_str).toLocaleString('en-US', { weekday: 'long', month: 'long', day: 'numeric'});
    var start_time = new Date(start_str).toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true});
    var end_time = new Date(end_str).toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true});
    var tz = new Date(start_str).toLocaleString('en-US', {timeZoneName: 'short'}).split(' ').pop();
    return day + ", " + start_time + " - " + end_time + " " + tz;
}
</script>

<div class="faq-accordion mt-4" id="schedule-accordion">
<div class="row">
    <div class="col-md-6" data-aos="fade-right">
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse0" aria-expanded="true" aria-controls="schedule-collapse0">
                <h5 class="my-0" style="display: inline-block;">Week 0 (Now - Oct 1<sup>st</sup>)</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse0" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <ul>
                        <li>Once you <a href="#pricing">apply</a> to the course, we will approve it and you'll receive a link to our Stripe checkout page.</li>
                        <li>You'll receive instructions to join our private community forum and introduce yourself to the cohort.</li>
                        <li>You'll receive the assignments and deliverables for Week 1.</li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse1" aria-expanded="true" aria-controls="schedule-collapse1">
                <h5 class="my-0" style="display: inline-block;">Week 1 (Oct 1<sup>st</sup> - Oct 7<sup>th</sup>)</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse1" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <ul>
                        <li><b>Individually: </b>🎨&nbsp; Design + 🔢&nbsp; Data lessons</li>
                        <li><b>Q&A sessions: </b>
                            <ul>
                                <li><span id="mon-1"></span></li>
                                <script>
                                    document.getElementById("mon-1").textContent = get_time(start_str="2022-10-03T08:00-07:00", end_str="2022-10-03T09:00-07:00")
                                </script>
                                <li><span id="wed-1"></span></li>
                                <script>
                                    document.getElementById("wed-1").textContent = get_time(start_str="2022-10-05T16:00-07:00", end_str="2022-10-05T17:00-07:00")
                                </script>
                                <li><span id="fri-1"></span></li>
                                <script>
                                    document.getElementById("fri-1").textContent = get_time(start_str="2022-10-07T12:00-07:00", end_str="2022-10-07T13:00-07:00")
                                </script>
                            </ul>
                        </li>
                        <li><b>Cohort reading: </b> Assigned (discussion next week)</li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse2" aria-expanded="true" aria-controls="schedule-collapse2">
                <h5 class="my-0" style="display: inline-block;">Week 2 (Oct 8th<sup>th</sup> - Oct 14<sup>th</sup>)</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse2" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <ul>
                        <li><b>Individually: </b>📈&nbsp; Modeling lessons</li>
                        <li><b>Q&A sessions: </b>
                            <ul>
                                <li><span id="mon-2"></span></li>
                                <script>
                                    document.getElementById("mon-2").textContent = get_time(start_str="2022-10-10T08:00-07:00", end_str="2022-10-10T09:00-07:00")
                                </script>
                                <li><span id="wed-2"></span></li>
                                <script>
                                    document.getElementById("wed-2").textContent = get_time(start_str="2022-10-12T16:00-07:00", end_str="2022-10-12T17:00-07:00")
                                </script>
                                <li><span id="fri-2"></span></li>
                                <script>
                                    document.getElementById("fri-2").textContent = get_time(start_str="2022-10-14T12:00-07:00", end_str="2022-10-14T13:00-07:00")
                                </script>
                            </ul>
                        </li>
                        <li><b>Cohort reading: </b>
                            <ul>
                                <li><span id="tue-2"></span></li>
                                <script>
                                    document.getElementById("tue-2").textContent = get_time(start_str="2022-10-11T08:00-07:00", end_str="2022-10-11T09:00-07:00")
                                </script>
                                <li><span id="thu-2"></span></li>
                                <script>
                                    document.getElementById("thu-2").textContent = get_time(start_str="2022-10-13T16:00-07:00", end_str="2022-10-13T17:00-07:00")
                                </script>
                            </ul>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse3" aria-expanded="true" aria-controls="schedule-collapse3">
                <h5 class="my-0" style="display: inline-block;">Week 3 (Oct 15<sup>th</sup> - Oct 21<sup>st</sup>)</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse3" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <ul>
                        <li><b>Individually: </b>💻&nbsp; Developing lessons</li>
                        <li><b>Q&A sessions: </b>
                            <ul>
                                <li><span id="mon-3"></span></li>
                                <script>
                                    document.getElementById("mon-3").textContent = get_time(start_str="2022-10-17T08:00-07:00", end_str="2022-10-17T09:00-07:00")
                                </script>
                                <li><span id="wed-3"></span></li>
                                <script>
                                    document.getElementById("wed-3").textContent = get_time(start_str="2022-10-19T16:00-07:00", end_str="2022-10-19T17:00-07:00")
                                </script>
                                <li><span id="fri-3"></span></li>
                                <script>
                                    document.getElementById("fri-3").textContent = get_time(start_str="2022-10-21T12:00-07:00", end_str="2022-10-21T13:00-07:00")
                                </script>
                            </ul>
                        </li>
                        <li><b>Cohort reading: </b> Assigned (discussion next week)</li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse4" aria-expanded="true" aria-controls="schedule-collapse4">
                <h5 class="my-0" style="display: inline-block;">Week 4 (Oct 22<sup>nd</sup> - Oct 28<sup>th</sup>)</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse4" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <ul>
                        <li><b>Individually: </b>📦&nbsp; Serving lessons</li>
                        <li><b>Q&A sessions: </b>
                            <ul>
                                <li><span id="mon-4"></span></li>
                                <script>
                                    document.getElementById("mon-4").textContent = get_time(start_str="2022-10-24T08:00-07:00", end_str="2022-10-24T09:00-07:00")
                                </script>
                                <li><span id="wed-4"></span></li>
                                <script>
                                    document.getElementById("wed-4").textContent = get_time(start_str="2022-10-26T16:00-07:00", end_str="2022-10-26T17:00-07:00")
                                </script>
                                <li><span id="fri-4"></span></li>
                                <script>
                                    document.getElementById("fri-4").textContent = get_time(start_str="2022-10-28T12:00-07:00", end_str="2022-10-28T13:00-07:00")
                                </script>
                            </ul>
                        </li>
                        <li><b>Cohort reading: </b>
                            <ul>
                                <li><span id="tue-4"></span></li>
                                <script>
                                    document.getElementById("tue-4").textContent = get_time(start_str="2022-10-25T08:00-07:00", end_str="2022-10-25T09:00-07:00")
                                </script>
                                <li><span id="thu-4"></span></li>
                                <script>
                                    document.getElementById("thu-4").textContent = get_time(start_str="2022-10-27T16:00-07:00", end_str="2022-10-27T17:00-07:00")
                                </script>
                            </ul>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse5" aria-expanded="true" aria-controls="schedule-collapse5">
                <h5 class="my-0" style="display: inline-block;">Week 5 (Oct 29<sup>th</sup> - Nov 4<sup>th</sup>)</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse5" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <ul>
                        <li><b>Individually: </b>✅&nbsp; Testing lessons</li>
                        <li><b>Q&A sessions: </b>
                            <ul>
                                <li><span id="mon-5"></span></li>
                                <script>
                                    document.getElementById("mon-5").textContent = get_time(start_str="2022-10-31T08:00-07:00", end_str="2022-10-31T09:00-07:00")
                                </script>
                                <li><span id="wed-5"></span></li>
                                <script>
                                    document.getElementById("wed-5").textContent = get_time(start_str="2022-11-02T16:00-07:00", end_str="2022-11-02T17:00-07:00")
                                </script>
                                <li><span id="fri-5"></span></li>
                                <script>
                                    document.getElementById("fri-5").textContent = get_time(start_str="2022-11-04T12:00-07:00", end_str="2022-11-04T13:00-07:00")
                                </script>
                            </ul>
                        </li>
                        <li><b>Cohort reading: </b> Assigned (discussion next week)</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-6" data-aos="fade-right">
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse6" aria-expanded="true" aria-controls="schedule-collapse6">
                <h5 class="my-0" style="display: inline-block;">Week 6 (Nov 5<sup>th</sup> - Nov 11<sup>th</sup>)</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse6" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <ul>
                        <li><b>Individually: </b>♻️&nbsp; Reproducibility lessons</li>
                        <li><b>Q&A sessions: </b>
                            <ul>
                                <li><span id="mon-6"></span></li>
                                <script>
                                    document.getElementById("mon-6").textContent = get_time(start_str="2022-11-07T08:00-08:00", end_str="2022-11-07T09:00-08:00")
                                </script>
                                <li><span id="wed-6"></span></li>
                                <script>
                                    document.getElementById("wed-6").textContent = get_time(start_str="2022-11-09T16:00-08:00", end_str="2022-11-09T17:00-08:00")
                                </script>
                                <li><span id="fri-6"></span></li>
                                <script>
                                    document.getElementById("fri-6").textContent = get_time(start_str="2022-11-11T12:00-08:00", end_str="2022-11-11T13:00-08:00")
                                </script>
                            </ul>
                        </li>
                        <li><b>Cohort reading: </b>
                            <ul>
                                <li><span id="tue-6"></span></li>
                                <script>
                                    document.getElementById("tue-6").textContent = get_time(start_str="2022-11-08T08:00-08:00", end_str="2022-11-08T09:00-08:00")
                                </script>
                                <li><span id="thu-6"></span></li>
                                <script>
                                    document.getElementById("thu-6").textContent = get_time(start_str="2022-11-10T16:00-08:00", end_str="2022-11-10T17:00-08:00")
                                </script>
                            </ul>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse7" aria-expanded="true" aria-controls="schedule-collapse7">
                <h5 class="my-0" style="display: inline-block;">Week 7 (Nov 12<sup>th</sup> - Nov 18<sup>th</sup>): OFF</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse7" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <ul>
                        <li>We're off this and next week for Thanksgiving break. Use this time to catch-up if you have fallen behind on the weekly deliverables and continue to ask and answer questions in the community!</li>
                        <li>A (longer) reading will be assigned to read over break and be discussed once we're back.</li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse8" aria-expanded="true" aria-controls="schedule-collapse8">
                <h5 class="my-0" style="display: inline-block;">Week 8 (Nov 19<sup>th</sup> - Nov 25<sup>th</sup>): OFF</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse8" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <div class="card-body">
                        <ul>
                            <li>We're off this and next week for Thanksgiving break. Use this time to catch-up if you have fallen behind on the weekly deliverables and continue to ask and answer questions in the community!</li>
                            <li>Continue with last week's (longer) reading assignment that will be discussed once we're back next week.</li>
                        </ul>
                </div>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse9" aria-expanded="true" aria-controls="schedule-collapse9">
                <h5 class="my-0" style="display: inline-block;">Week 9 (Nov 26<sup>th</sup> - Dec 2<sup>nd</sup>)</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse9" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <ul>
                        <li><b>Individually: </b>🚀&nbsp; Production lessons</li>
                        <li><b>Q&A sessions: </b>
                            <ul>
                                <li><span id="wed-9"></span></li>
                                <script>
                                    document.getElementById("wed-9").textContent = get_time(start_str="2022-11-30T16:00-08:00", end_str="2022-11-30T17:00-08:00")
                                </script>
                                <li><span id="mon-9"></span></li>
                                <script>
                                    document.getElementById("mon-9").textContent = get_time(start_str="2022-12-01T08:00-08:00", end_str="2022-12-01T09:00-08:00")
                                </script>
                                <li><span id="fri-9"></span></li>
                                <script>
                                    document.getElementById("fri-9").textContent = get_time(start_str="2022-12-02T12:00-08:00", end_str="2022-12-02T13:00-08:00")
                                </script>
                            </ul>
                        </li>
                        <li><b>Cohort reading: </b>
                            <ul>
                                <li><span id="tue-9"></span></li>
                                <script>
                                    document.getElementById("tue-9").textContent = get_time(start_str="2022-11-29T08:00-08:00", end_str="2022-11-29T09:00-08:00")
                                </script>
                                <li><span id="thu-9"></span></li>
                                <script>
                                    document.getElementById("thu-9").textContent = get_time(start_str="2022-12-01T16:00-08:00", end_str="2022-12-01T17:00-08:00")
                                </script>
                            </ul>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse10" aria-expanded="true" aria-controls="schedule-collapse10">
                <h5 class="my-0" style="display: inline-block;">Week 10 (Dec 3<sup>rd</sup> - Dec 9<sup>th</sup>)</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse10" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <ul>
                        <li><b>Individually: </b>⎈&nbsp; Data engineering lessons</li>
                        <li><b>Q&A sessions: </b>
                            <ul>
                                <li><span id="mon-10"></span></li>
                                <script>
                                    document.getElementById("mon-10").textContent = get_time(start_str="2022-12-05T08:00-08:00", end_str="2022-12-05T09:00-08:00")
                                </script>
                                <li><span id="wed-10"></span></li>
                                <script>
                                    document.getElementById("wed-10").textContent = get_time(start_str="2022-12-07T16:00-08:00", end_str="2022-12-07T17:00-08:00")
                                </script>
                                <li><span id="fri-10"></span></li>
                                <script>
                                    document.getElementById("fri-10").textContent = get_time(start_str="2022-12-09T12:00-08:00", end_str="2022-12-09T13:00-08:00")
                                </script>
                            </ul>
                        </li>
                        <li><b>Cohort reading: </b> Assigned (discussion next week)</li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header collapsed" id="heading1" data-toggle="collapse" data-target="#schedule-collapse11" aria-expanded="true" aria-controls="schedule-collapse11">
                <h5 class="my-0" style="display: inline-block;">Week 11 (Dec 10<sup>th</sup> - Dec 16<sup>th</sup>)</h5>
                <span class="faq-toggle" aria-hidden="true" style="float:right;"></span>
            </div>
            <div id="schedule-collapse11" class="collapse" aria-labelledby="heading1" data-parent="#schedule-accordion">
                <div class="card-body">
                    <ul>
                        <li><b>Individually: </b>Conclusion</li>
                        <li><b>Q&A sessions: </b>
                            <ul>
                                <li><span id="mon-11"></span></li>
                                <script>
                                    document.getElementById("mon-11").textContent = get_time(start_str="2022-12-12T08:00-08:00", end_str="2022-12-12T09:00-08:00")
                                </script>
                                <li><span id="wed-11"></span></li>
                                <script>
                                    document.getElementById("wed-11").textContent = get_time(start_str="2022-12-14T16:00-08:00", end_str="2022-12-14T17:00-08:00")
                                </script>
                                <li><span id="fri-11"></span></li>
                                <script>
                                    document.getElementById("fri-11").textContent = get_time(start_str="2022-12-16T12:00-08:00", end_str="2022-12-16T13:00-08:00")
                                </script>
                            </ul>
                        </li>
                        <li><b>Cohort reading: </b>
                            <ul>
                                <li><span id="tue-11"></span></li>
                                <script>
                                    document.getElementById("tue-11").textContent = get_time(start_str="2022-12-13T08:00-08:00", end_str="2022-12-13T09:00-08:00")
                                </script>
                                <li><span id="thu-11"></span></li>
                                <script>
                                    document.getElementById("thu-11").textContent = get_time(start_str="2022-12-15T16:00-08:00", end_str="2022-12-15T17:00-08:00")
                                </script>
                            </ul>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</div>
</div>