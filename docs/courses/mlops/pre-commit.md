---
template: lesson.html
title: "Pre-commit"
description: Using the pre-commit git hooks to ensure checks before committing.
keywords: pre-commit, git hooks, git, versioning, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning, great expectations
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Intuition

Before performing a commit to our local repository, there are a lot of items on our mental todo list, ranging from styling, formatting, testing, etc. And it's very easy to forget some of these steps, especially when we want to "push to quick fix". To help us manage all these important steps, we can use pre-commit hooks, which will automatically be triggered when we try to perform a commit.

!!! note
    Though we can add these checks directly in our CI/CD pipeline (ex. via GitHub actions), it's significantly faster to validate our commits before pushing to our remote host and waiting to see what needs to be fixed before submitting yet another PR.

## Application

We'll be using the [Pre-commit](https://pre-commit.com/){:target="_blank"} framework to help us automatically perform important checks via hooks when we make a commit.

```bash
# Install pre-commit
pip install pre-commit
pre-commit install
```

### Config

We define our pre-commit hooks via a [.pre-commit-config.yaml](https://github.com/GokuMohandas/MLOps/blob/main/.pre-commit-config.yaml){:target="_blank"} configuration file. We can either create our yaml configuration from scratch or use the pre-commit CLI to create a sample configuration which we can add to.

```bash
# Simple config
pre-commit sample-config > .pre-commit-config.yaml
cat .pre-commit-config.yaml
```

```yaml linenums="1"
# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v3.2.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
```

### Built-in hooks

Inside the sample configuration, we can see that pre-commit has added some default hooks from it's repository. It specifies the location of the repository, version as well as the specific hook ids to use. We can read about the function of these hooks and add even more by exploring pre-commit's [built-in hooks](https://github.com/pre-commit/pre-commit-hooks){:target="_blank"}. Many of them accept arguments such as `maxkb` (max kilobytes) for the `check-added-large-files` hook. We can easily add argument to our configuration file like so:

```yaml
# Inside .pre-commit-config.yaml
...
-   id: check-added-large-files
    args: ['--maxkb=1000']
...
```

!!! note
    Be sure to explore the many other [built-in hooks](https://github.com/pre-commit/pre-commit-hooks){:target="_blank"} because there are some really useful ones that we use in our project. For example, `check-merge-conflict` to see if there are any lingering merge conflict strings or `detect-aws-credentials` if we accidently left our credentials exposed in a file, and so much more.

And we can also exclude certain files from being processed by the hooks by using the optional *exclude* key.
```yaml
# Inside .pre-commit-config.yaml
...
-   id: check-yaml
    exclude: "mkdocs.yml"
...
```

There are many other [optional keys](https://pre-commit.com/#pre-commit-configyaml---hooks){:target="_blank"} we can configure for each hook ID.


### Custom hooks

Besides pre-commit's built-in hooks, there are also many custom, 3rd party [popular hooks](https://pre-commit.com/hooks.html){:target="_blank"} that we can choose from. For example, if we want to apply formatting checks with Black as a hook, we can leverage Black's pre-commit hook.

```yaml
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

### Local hooks

We can also create our own local hooks without configuring a separate .pre-commit-hooks.yaml. Here we're defining two pre-commit hooks, `test-non-training` and `clean`, to run some commands that we've defined in our Makefile. Similarly, we can run any entry command with arguments to create hooks very quickly.

```yaml
# Inside .pre-commit-config.yaml
...
- repo: local
  hooks:
    - id: test-non-training
      name: test-non-training
      entry: make
      args: ["test-non-training"]
      language: system
      pass_filenames: false
    - id: clean
      name: clean
      entry: make
      args: ["clean"]
      language: system
      pass_filenames: false
```

### Commit

Our pre-commit hooks will automatically execute when we try to make a commit. We'll be able to see if each hook passed or failed and make any changes. If any of the hooks failed, we have to fix the corresponding file or in many instances, reformatting will occur automatically.

<div class="ai-center-all">
    <img width="650" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pre-commit/reformat.png" style="border-radius: 7px;">
</div>

Once we've made or approved the changes, we can commit again to ensure that all hooks are passed.

<div class="ai-center-all">
    <img width="650" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pre-commit/commit.png" style="border-radius: 7px;">
</div>

### Run

Though pre-commit hooks are meant to run before (pre) a commit, we can manually trigger all or individual hooks on all or a set of files.

```bash
# Run
pre-commit run --all-files  # run all hooks on all files
pre-commit run <HOOK_ID> --all-files # run one hook on all files
pre-commit run --files <PATH_TO_FILE>  # run all hooks on a file
pre-commit run <HOOK_ID> --files <PATH_TO_FILE> # run one hook on a file
```

### Skip

It is highly not recommended to skip running any of the pre-commit hooks because they are there for a reason. But for some highly urgent, world saving commits, we can use the no-verify flag.

```bash
# Commit without hooks
git commit -m <MESSAGE> --no-verify
```

### Update

In our [.pre-commit-config.yaml](https://github.com/GokuMohandas/MLOps/blob/main/.pre-commit-config.yaml){:target="_blank"} configuration files, we've had to specify the versions for each of the repositories so we can use their latest hooks. Pre-commit has an autoupdate CLI command which will update these versions as they become available.

```bash
# Autoupdate
pre-commit autoupdate
```

We can also add this command to our Makefile to execute when a development environment is created so everything is up-to-date.

```yaml hl_lines="7"
# Makefile
...
.PHONY: install-dev
install-dev:
	python -m pip install -e ".[dev]" --no-cache-dir
	pre-commit install
	pre-commit autoupdate
...
```

<!-- Citation -->
{% include "cite.md" %}