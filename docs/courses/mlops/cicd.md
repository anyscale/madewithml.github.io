---
template: lesson.html
title: "CI/CD for Machine Learning"
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
    <img width="700" src="/static/images/mlops/cicd/workflows.png">
</div>

GitHub Actions has the added advantage of integrating really well with GitHub and since all of our work is versioned there, we can easily create workflows based on GitHub events (push, PR, release, etc.). GitHub Actions also has a rich marketplace full of workflows that we can use for our own project. And, best of all, GitHub Actions is [free for public repositories](https://docs.github.com/en/github/setting-up-and-managing-billing-and-payments-on-github/about-billing-for-github-actions){:target="_blank"}.

## Components

We'll learn about GitHub Actions by understanding the components that compose an Action. These components abide by a specific [workflow syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions){:target="_blank"} which can be extended with the appropriate [context and expression syntax](https://docs.github.com/en/actions/reference/context-and-expression-syntax-for-github-actions){:target="_blank"}.

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/cicd/actions.png">
</div>

### Workflows

With GitHub Actions, we are creating automatic **workflows** to do something for us. We'll start by creating a .github/workflows directory to organize all of our workflows.

```bash
mkdir -p .github/workflows
touch .github/workflows/testing.yml
touch .github/workflows/documentation.yml
```

Each workflow file will contain the specific instructions for that action. For example, this [testing workflow](https://github.com/GokuMohandas/MLOps/blob/main/.github/workflows/testing.yml){:target="_blank"} is responsible for conducting tests on our code base. We can specify the name of our workflow at the top of our YAML file.

```yaml linenums="1"
# .github/workflows/testing.yml
name: testing
```

### Events

Workflows are triggered by an **event**, which can be something that occurs on a schedule ([cron](https://pubs.opengroup.org/onlinepubs/9699919799/utilities/crontab.html){:target="_blank"}), webhook or manually. In our application, we'll be using the [push](https://docs.github.com/en/actions/reference/events-that-trigger-workflows#push){:target="_blank"} and [pull request](https://docs.github.com/en/actions/reference/events-that-trigger-workflows#pull_request){:target="_blank"} webhook events to run the testing workflow when someone directly pushes or submits a PR to the main branch.

```yaml linenums="1"
# .github/workflows/testing.yml
on:
  push:
    branches:
    - main
    - master
  pull_request:
    branches:
    - main
    - master
```

> Be sure to check out the [complete list](https://docs.github.com/en/actions/reference/events-that-trigger-workflows){:target="_blank"} of the different events that can trigger a workflow.

### Jobs

Once the event is triggered, a set of **jobs** run on a [**runner**](https://github.com/actions/runner){:target="_blank"}, which is the application that runs the job using a specific operating system. Our first (and only) job is `test-code` which runs on the latest version of ubuntu.

```yaml linenums="1"
# .github/workflows/testing.yml
jobs:
  test-code:
    runs-on: ubuntu-latest
```

> Jobs run in parallel but if you need to create dependent jobs, where if a particular job fails all it's dependent jobs will be skipped, then be sure to use the [needs](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#jobsjob_idneeds){:target="_blank"} key. One a similar note, we can also [share](https://docs.github.com/en/actions/learn-github-actions/essential-features-of-github-actions#sharing-data-between-jobs){:target="_blank"} data between jobs.

### Steps

Each job contains a series of **steps** which are executed in order. Each step has a name, as well as actions to use from the GitHub Action marketplace or commands we want to run. For the `test-code` job, the steps are to checkout the repo, install the necessary dependencies and run tests.

```yaml linenums="1"
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
        run: |
          python3 -m pip install -e ".[test]" --no-cache-dir
      - name: Execute tests
        run: pytest tests/tagifai --ignore tests/code/test_main.py --ignore tests/code/test_data.py
```

> We are only executing a subset of the tests here because we won't access to data or model artifacts when these tests are executed on GitHub's runners. However, if our blob storage and model registry are on the cloud, we can access them and perform all the tests. This will often involve using credentials to access these resources, which we can set as Action secrets (GitHub repository page > `Settings` > `Secrets`).

??? quote "View `.github/workflows/testing.yml`"
    ```yaml linenums="1"
    name: testing
    on:
    push:
        branches:
        - master
        - main
    pull_request:
        branches:
        - master
        - main
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
            run: |
            python -m pip install -e ".[test]" --no-cache-dir
        - name: Execute tests
            run: pytest tests/tagifai --ignore tests/tagifai/test_main.py --ignore tests/tagifai/test_data.py
    ```

Notice that one of our steps is to [cache](https://docs.github.com/en/actions/guides/caching-dependencies-to-speed-up-workflows){:target="_blank"} the entire Python environment with a specific key. This will significantly speed up the time required to run our Action the next time as long as the key remains unchanged (same python location, setup.py and requirements.txt).

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/cicd/cache.png">
</div>

Our other workflow is responsible for automatically generating and deploying our mkdocs documentation. The "Deploy documentation" step below will create/update a new branch in our repository called [gh-pages](https://github.com/GokuMohandas/MLOps/tree/gh-pages){:target="_blank"} which will have the generation UI files for our documentation. We can deploy this branch as a GitHub pages website by going to `Settings` > `Pages` and setting the source branch to `gh-pages` and folder to `/root` > `Save`. This will generate the public URL for our documentation and it will automatically update every time our workflow runs after each PR.

```yaml linenums="1"
# .github/workflows/documentation.yml
name: documentation
...
jobs:
  build-docs:
      ...
      - name: Deploy documentation
        run: mkdocs gh-deploy --force
```

??? quote "View `.github/workflows/documentation.yml`"
    ```yaml linenums="1"
    name: documentation
    on:
    push:
        branches:
        - master
        - main
    pull_request:
        branches:
        - master
        - main
    jobs:
    build-docs:
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
            run: |
            python -m pip install -e ".[docs]" --no-cache-dir
        - name: Deploy documentation
            run: mkdocs gh-deploy --force
    ```

> We can also generate [private documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site){:target="_blank"} for private repositories and even host it on a [custom domain](https://docs.github.com/en/github/working-with-github-pages/configuring-a-custom-domain-for-your-github-pages-site){:target="_blank"}.

## Runs

Recall that workflows will be triggered when certain events occur. For example, our testing workflow will initiate on a push or PR to the main branch. We can see the workflow's runs (current and previous) on the *Actions* tab on our repository page. And if we click on a specific run, we can view the all the steps and their outputs as well. We can also set branch protection rules (GitHub repository page > `Settings` > `Branches`) to ensure that these workflow runs are all successful before we can merge to the main branch.

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/cicd/results.png">
</div>

> While there are methods, such as [act](https://github.com/nektos/act){:target="_blank"}, to run and test workflows locally, many of them are not stable enough for reliable use.

## Serving

There are a wide variety of GitHub actions available for deploying and serving our ML applications after all the integration tests have passed. Most of them will require that we have a Dockerfile defined that will load and launch our service with the appropriate artifacts. Read more about ML deployment infrastructure in our [lesson](infrastructure.md){:target="_blank"}.

- [AWS EC2](https://github.com/aws-actions){:target="_blank"}, [Google Compute Engine](https://github.com/google-github-actions){:target="_blank"}, [Azure VM](https://github.com/Azure/actions){:target="_blank"}, etc.
- container orchestration services such as [AWS ECS](https://github.com/aws-actions/amazon-ecs-deploy-task-definition){:target="_blank"}  or [Google Kubernetes Engine](https://github.com/google-github-actions/setup-gcloud/tree/master/example-workflows/gke){:target="_blank"}
- serverless options such as [AWS Lambda](https://github.com/marketplace/actions/aws-lambda-deploy){:target="_blank"} or [Google Cloud Functions](https://github.com/google-github-actions/deploy-cloud-functions){:target="_blank"}.

> If we want to deploy and serve multiple models at a time, it's highly recommended to use a purpose-built [model server](api.md#model-server){:target="_blank"} to seamlessly inspect, update, serve, rollback, etc. multiple versions of models.

The specific deployment method we use it entirely up dependent on the application, team, existing infrastructure, etc. The key component is that we are able to update our application when all the integration tests pass without having to manually intervene for deployment.

## Marketplace

So what exactly are these actions that we're using from the marketplace? For example, our first step in the `test-code` job above is to checkout the repo using the [actions/checkout@v2](https://github.com/marketplace/actions/checkout){:target="_blank"} GitHub Action. The Action's link contains information about how to use it, scenarios, etc. The Marketplace has actions for a variety of needs, ranging from continuous deployment for various cloud providers, code quality checks, etc.

- [Great Expectations](https://github.com/marketplace/actions/great-expectations-data){:target="_blank"}: ensure that our GE checkpoints pass when any changes are made that could affect the data engineering pipelines. This action also creates a free GE dashboard with [Netlify](https://www.netlify.com/){:target="_blank"} that has the updated data docs.
- [Continuous ML](https://github.com/iterative/cml){:target="_blank"}: train, evaluate and monitor your ML models and generate a report summarizing the workflows. If you don't want to train offline, you can manually/auto trigger the training pipeline to run on cloud infrastructure (AWS/GCP) or [self hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners?learn=hosting_your_own_runners){:target="_blank"}.

> Don't restrict your workflows to only what's available on the Marketplace or single command operations. We can do things like include code coverage reports, deploy an updated Streamlit dashboard and attach it's URL to the PR, deliver (CD) our application to an AWS Lambda / EC2, etc.

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions){:target="_blank"}
- [Act - Run your GitHub Actions Locally](https://github.com/nektos/act){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}