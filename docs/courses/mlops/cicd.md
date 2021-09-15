---
template: lesson.html
title: "CI/CD Workflows"
description: Using workflows to establish continuous integration and delivery pipelines to reliably iterate on our application.
keywords: ci/cd, github actions, devops, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning, great expectations
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Intuition

Continuous integration (CI) allows our team to develop, test and integrate code in a structured fashion. This allows the team to more confidently and frequently develop since their work will be properly integrated. Continuous delivery (CD) is responsible for delivering our integrated code to a variety of applications that are dependent on it. With CI/CD pipelines, we can develop and deploy knowing that our systems can quickly adapt and work as intended.

## GitHub Actions

There are many tooling options for when it comes to creating our CI/CD pipelines, such as [Jenkins](https://www.jenkins.io/){:target="_blank"}, [TeamCity](https://www.jetbrains.com/teamcity/){:target="_blank"}, [CircleCI](https://circleci.com/){:target="_blank"} and many others. However, we're going to use [GitHub Actions](https://docs.github.com/en/actions){:target="_blank"} to create automatic workflows to setup our CI/CD pipelines.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/cicd/workflows.png">
</div>

GitHub Actions has the added advantage of integrating really well with GitHub and since all of our work is versioned there, we can easily create workflows based on GitHub events (push, PR, release, etc.). GitHub Actions also has a rich marketplace full of workflows that we can use for our own project. And, best of all, GitHub Actions is [free](https://docs.github.com/en/github/setting-up-and-managing-billing-and-payments-on-github/about-billing-for-github-actions){:target="_blank"} for public repositories and actions using self-hosted runners (running workflows on your own hardware or on the cloud).

## Components

We'll learn about GitHub Actions by understanding the components that compose an Action. These components abide by a specific [workflow syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions){:target="_blank"} which can be extended with the appropriate [context and expression syntax](https://docs.github.com/en/actions/reference/context-and-expression-syntax-for-github-actions){:target="_blank"}.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/cicd/actions.png">
</div>

### Workflows

With GitHub Actions, we are creating automatic **workflows** to do something for us. For example, this [testing workflow](https://github.com/GokuMohandas/MLOps/blob/main/.github/workflows/testing.yml){:target="_blank"} is responsible for conducting tests on our code base. We can specify the name of our workflow at the top of our YAML file.

```yaml
# .github/workflows/testing.yml
name: testing
```

### Events

Workflows are triggered by an **event**, which can be something that occurs on a schedule ([cron](https://pubs.opengroup.org/onlinepubs/9699919799/utilities/crontab.html){:target="_blank"}), webhook or manually. In our application, we'll be using the [push](https://docs.github.com/en/actions/reference/events-that-trigger-workflows#push){:target="_blank"} and [pull request](https://docs.github.com/en/actions/reference/events-that-trigger-workflows#pull_request){:target="_blank"} webhook events to run the testing workflow when someone directly pushes or submits a PR to the main branch.

```yaml
# .github/workflows/testing.yml
on:
  push:
    branches:
    - main
  pull_request:
    branches:
    - main
```

!!! note
    Be sure to check out the [complete list](https://docs.github.com/en/actions/reference/events-that-trigger-workflows){:target="_blank"} of the different events that can trigger a workflow.

### Jobs

Once the event is triggered, a set of **jobs** run on a [**runner**](https://github.com/actions/runner){:target="_blank"}, which is the application that runs the job using a specific operating system. Our first (and only) job is `test-code` which runs on the latest version of ubuntu.

```yaml
# .github/workflows/testing.yml
jobs:
  test-code:
    runs-on: ubuntu-latest
```

!!! note
    Jobs run in parallel but if you need to create dependent jobs, where if a particular job fails all it's dependent jobs will be skipped, then be sure to use the [needs](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#jobsjob_idneeds){:target="_blank"} key. One a similar note, we can also [share](https://docs.github.com/en/actions/learn-github-actions/essential-features-of-github-actions#sharing-data-between-jobs){:target="_blank"} data between jobs.

### Steps

Each job contains a series of **steps** which are executed in order. Each step has a name, as well as actions to use from the GitHub Action marketplace or commands we want to run. For the `test-code` job, the steps are to checkout the repo, install the necessary dependencies and run tests.

```yaml
# .github/workflows/testing.yml
jobs:
  test-code:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.7.10
      - name: Caching
        uses: actions/cache@v2
        with:
          path: $/{/{ env.pythonLocation /}/}
          key: $/{/{ env.pythonLocation /}/}-$/{/{ hashFiles('setup.py') /}/}-$/{/{ hashFiles('requirements.txt') /}/}
      - name: Install dependencies
        run: python -m pip install -e ".[test]" --no-cache-dir
      - name: Run tests
        run: pytest tests/tagifai --cov tagifai --cov-report html
```

!!! note
    Notice that one of our steps is to [cache](https://docs.github.com/en/actions/guides/caching-dependencies-to-speed-up-workflows){:target="_blank"} the entire Python environment with a specific key. This will significantly speed up the time required to run our Action the next time as long as the key remains unchanged (same python location, setup.py and requirements.txt).

    <div class="ai-center-all">
        <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/cicd/cache.png">
    </div>

## Runs

Recall that workflows will be triggered when certain events occur. For example, our testing workflow will initiate on a push or PR to the main branch. We can see the workflow's runs (current and previous) on the *Actions* tab on our repository page. And if we click on a specific run, we can view the all the steps and their outputs as well.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/cicd/results.png">
</div>

However, when we're first creating a GitHub Action, we may not get our workflow to run so smoothly. Instead of creating multiple commits (if you do, at least squash them together), we can use [act](https://github.com/nektos/act){:target="_blank"} to run and test workflows locally. Act executes the jobs defined in our workflows by spinning up a container with an image that is very similar to the image that GitHub's runners use. It has the same environment specifications, variables, etc. and you can inspect both the images and running containers with the respective docker commands (docker images, docker ps).

<div class="animated-code">

    ```console
    # List of jobs
    $ act -l
    ID          Stage  Name
    build-docs  0      build-docs
    test-code   0      test-code
    ```

</div>
<script src="../../../static/js/termynal.js"></script>

With act, we can run all the jobs, a specific job, jobs for a specific event (ex. PR), etc.

```bash
# Running GitHub Actions locally
act  # Run default PUSH event
act -l  # list all jobs
act pull_request  # PR event
act -j test-code  # specific job
```

<pre class="output">
<span style="color: #0071CF;">[testing/test-code]</span> 🚀  Start image=catthehacker/ubuntu:act-latest
<span style="color: #0071CF;">[testing/test-code]</span>   🐳  docker run image=catthehacker/ubuntu:act-latest platform=linux/amd64 entrypoint=["/usr/bin/tail" "-f" "/dev/null"] cmd=[]
<span style="color: #0071CF;">[testing/test-code]</span>   🐳  docker cp src=/Users/goku/Documents/madewithml/mlops/. dst=/Users/goku/Documents/madewithml/mlops
<span style="color: #0071CF;">[testing/test-code]</span> ⭐  Run Checkout repo
<span style="color: #0071CF;">[testing/test-code]</span>   ✅  Success - Checkout repo
...
<span style="color: #0071CF;">[testing/test-code]</span> ⭐  Run Execute tests
<span style="color: #0071CF;">[testing/test-code]</span>   ✅  Success - Execute tests
</pre>

!!! note
    While act is able to very closely replicate GitHub's runners there are still a few inconsistencies. For example [caching](https://github.com/nektos/act/issues/329){:target="_blank"} still needs to be figured out with a small HTTP server when the local container is spun up. So if we have a lot of requirements, it might be faster just to experience using GitHub's runners and squashing the commits once we get the workflow to run.

## Serving

There are a wide variety of GitHub actions available for deploying and serving our ML applications after all the integration tests have passed. Most of them will require that we have a Dockerfile defined that will load and launch our service with the appropriate artifacts. Read more about ML deployment infrastructure in our [lesson](infrastructure.md){:target="_blank"}.

- [AWS EC2](https://github.com/aws-actions), [Google Compute Engine](https://github.com/google-github-actions), [Azure VM](https://github.com/Azure/actions), etc.
- container orchestration services such as [AWS ECS](https://github.com/aws-actions/amazon-ecs-deploy-task-definition)  or [Google Kubernetes Engine](https://github.com/google-github-actions/setup-gcloud/tree/master/example-workflows/gke)
- serverless options such as [AWS Lambda](https://github.com/marketplace/actions/aws-lambda-deploy) or [Google Cloud Functions](https://github.com/google-github-actions/deploy-cloud-functions).

!!! note
    If we want to deploy and serve multiple models at a time, it's highly recommended to use a purpose-built model server such as [MLFlow](https://mlflow.org/){:target="_blank"}, [TorchServe](https://pytorch.org/serve/){:target="_blank"}, [RedisAI](https://oss.redislabs.com/redisai/){:target="_blank"} or [Nvidia's Triton](https://developer.nvidia.com/nvidia-triton-inference-server){:target="_blank"} inference server. These servers have a registry with an API layer to seamlessly inspect, update, serve, rollback, etc. multiple versions of models.

The specific deployment method we use it entirely up dependent on the application, team, existing infrastructure, etc. The key component is that we are able to update our application when all the integration tests pass without having to manually intervene for deployment.

## Marketplace

So what exactly are these actions that we're using from the marketplace? For example, our first step in the `test-code` job above is to checkout the repo using the [actions/checkout@v2](https://github.com/marketplace/actions/checkout){:target="_blank"} GitHub Action. The Action's link contains information about how to use it, scenarios, etc.

The Marketplace has actions for a variety of needs, ranging from continuous deployment for various cloud providers, code quality checks, etc. Below are a few GitHub Actions that we highly recommend.

- [Great Expectations](https://github.com/marketplace/actions/great-expectations-data){:target="_blank"}: ensure that our GE checkpoints pass when any changes are made that could affect the data engineering pipelines. This action also creates a free GE dashboard with [Netlify](https://www.netlify.com/){:target="_blank"} that has the updated data docs.
- [Continuous ML](https://github.com/iterative/cml){:target="_blank"}: train, evaluate and monitor your ML models and generate a report summarizing the findings. I personally use this GitHub Action for automatic training jobs on cloud infrastructure (AWS/GCP) or [self hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners?learn=hosting_your_own_runners){:target="_blank"} when a change triggers the training pipeline, as opposed to working with [Terraform](https://www.terraform.io/){:target="_blank"}.

!!! note
    Don't restrict your workflows to only what's available on the Marketplace or single command operations. We can do things like include code coverage reports, deploy an updated Streamlit dashboard and attach it's URL to the PR, deliver (CD) our application to an AWS Lambda / EC2, etc.

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions){:target="_blank"}
- [Act - Run your GitHub Actions Locally](https://github.com/nektos/act){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}