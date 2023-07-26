---
template: lesson.html
title: "Pre-commit"
description: Using the pre-commit git hooks to ensure checks before committing.
keywords: pre-commit, git hooks, git, versioning, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
---

{% include "styles/lesson.md" %}

## Intuition

Before performing a commit to our local repository, there are a lot of items on our mental todo list, ranging from styling, formatting, testing, etc. And it's very easy to forget some of these steps, especially when we want to "push to quick fix". To help us manage all these important steps, we can use pre-commit hooks, which will automatically be triggered when we try to perform a commit. These hooks can ensure that certain rules are followed or specific actions are executed successfully and if any of them fail, the commit will be aborted.

## Installation

We'll be using the [Pre-commit](https://pre-commit.com/){:target="_blank"} framework to help us automatically perform important checks via hooks when we make a commit.

> We'll start by installing and autoupdating pre-commit (we only have to do this once).
```bash
pre-commit install
pre-commit autoupdate
```

## Config

We define our pre-commit hooks via a `.pre-commit-config.yaml` configuration file. We can either create our yaml configuration from scratch or use the pre-commit CLI to create a sample configuration which we can add to.

```bash
# Simple config
pre-commit sample-config > .pre-commit-config.yaml
cat .pre-commit-config.yaml
```

## Hooks

When it comes to creating and using hooks, we have several options to choose from.

### Built-in

Inside the sample configuration, we can see that pre-commit has added some default hooks from it's repository. It specifies the location of the repository, version as well as the specific hook ids to use. We can read about the function of these hooks and add even more by exploring pre-commit's [built-in hooks](https://github.com/pre-commit/pre-commit-hooks){:target="_blank"}. Many of them also have additional arguments that we can configure to customize the hook.

```yaml linenums="1"
# Inside .pre-commit-config.yaml
...
-   id: check-added-large-files
    args: ['--maxkb=1000']
    exclude: "notebooks"
...
```

> Be sure to explore the many other [built-in hooks](https://github.com/pre-commit/pre-commit-hooks){:target="_blank"} because there are some really useful ones that we use in our project. For example, `check-merge-conflict` to see if there are any lingering merge conflict strings or `detect-aws-credentials` if we accidentally left our credentials exposed in a file, and so much more.

And we can also exclude certain files from being processed by the hooks by using the optional *exclude* key. There are many other [optional keys](https://pre-commit.com/#pre-commit-configyaml---hooks){:target="_blank"} we can configure for each hook ID.

```yaml linenums="1"
# Inside .pre-commit-config.yaml
...
-   id: check-yaml
    exclude: "mkdocs.yml"
...
```

### Custom

Besides pre-commit's built-in hooks, there are also many custom, 3rd party [popular hooks](https://pre-commit.com/hooks.html){:target="_blank"} that we can choose from. For example, if we want to apply formatting checks with Black as a hook, we can leverage Black's pre-commit hook.

```yaml linenums="1"
# Inside .pre-commit-config.yaml
...
-   repo: https://github.com/psf/black
    rev: 20.8b1
    hooks:
    -   id: black
        args: []
        files: .
...
```

This specific hook is defined under a [.pre-commit-hooks.yaml](https://github.com/psf/black/blob/master/.pre-commit-hooks.yaml){:target="_blank"} inside Black's repository, as are other custom hooks under their respective package repositories.

### Local

We can also create our own local hooks without configuring a separate .pre-commit-hooks.yaml. Here we're defining two pre-commit hooks, `test-non-training` and `clean`, to run some commands that we've defined in our Makefile. Similarly, we can run any entry command with arguments to create hooks very quickly.

```yaml linenums="1"
# Inside .pre-commit-config.yaml
...
-   repo: local
    hooks:
    -   id: clean
        name: clean
        entry: make
        args: ["clean"]
        language: system
        pass_filenames: false
```

## Commit

Our pre-commit hooks will automatically execute when we try to make a commit. We'll be able to see if each hook passed or failed and make any changes. If any of the hooks fail, we have to fix the errors ourselves or, in many instances, reformatting will occur automatically.

<pre class="output">
check yaml..............................................<span style="color: #39BC70;">PASSED</span>
clean...................................................<span style="color: #F50071;">FAILED</span>
</pre>

In the event that any of the hooks failed, we need to `add` and `commit` again to ensure that all hooks are passed.

```bash
git add .
git commit -m <MESSAGE>
```

<div class="ai-center-all">
    <img width="650" src="/static/images/mlops/pre_commit/commit.png" style="border-radius: 7px;" alt="precommit">
</div>

## Run

Though pre-commit hooks are meant to run before (pre) a commit, we can manually trigger all or individual hooks on all or a set of files.

```bash
# Run
pre-commit run --all-files  # run all hooks on all files
pre-commit run <HOOK_ID> --all-files # run one hook on all files
pre-commit run --files <PATH_TO_FILE>  # run all hooks on a file
pre-commit run <HOOK_ID> --files <PATH_TO_FILE> # run one hook on a file
```

## Skip

It is highly not recommended to skip running any of the pre-commit hooks because they are there for a reason. But for some highly urgent, world saving commits, we can use the no-verify flag.

```bash
# Commit without hooks
git commit -m <MESSAGE> --no-verify
```

> Highly recommend **not** doing this because no commit deserves to be force pushed no matter how "small" your change was. If you accidentally did this and want to clear the cache, run `#!bash pre-commit run --all-files` and execute the commit message operation again.

## Update

In our `.pre-commit-config.yaml` configuration files, we've had to specify the versions for each of the repositories so we can use their latest hooks. Pre-commit has an autoupdate CLI command which will update these versions as they become available.

```bash
# Autoupdate
pre-commit autoupdate
```

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}