---
template: lesson.html
title: Makefiles
description: An automation tool that organizes commands for our application's processes.
keywords: makefile, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/follow/tree/makefile
---

{% include "styles/lesson.md" %}

## Intuition

We have just started and there are already so many different commands to keep track of. To help with this, we're going to use a [`Makefile`](https://opensource.com/article/18/8/what-how-makefile){:target="_blank"} which is a automation tool that organizes our commands. This makes it very easy for us to organize relevant commands as well as organize it for others who may be new to our application.

## Components

Inside our [Makefile](https://github.com/GokuMohandas/MLOps/tree/main/Makefile){:target="_blank"}, we can see a list of rules (help, venv, clean, etc.). These rules have a `target` which can sometimes have `prerequisites` that need to be met (can be other targets) and on the next line a ++tab++ followed by a `recipe` which specifies how to create the target.

```bash
# Makefile
target: prerequisites
<TAB> recipe
```

## Targets
We can execute any of the rules by typing `make <target>`:

<div class="animated-code">

    ```console
    # View all rules
    $ make help
    👉  Commands:
        venv   : creates development environment.
        style  : runs style formatting.
        clean  : cleans all unnecessary files.

    # Make a target
    $ make venv
    python3 -m venv venv
    ...
    ```

</div>
<script src="../../../static/js/termynal.js"></script>

## PHONY
A Makefile is called as such because traditionally the `targets` are supposed to be files we can make. However, Makefiles are also commonly used as command shortcuts which can lead to confusion when a file with a certain name exists and a Makefile rule with the same name exists! For example if you a directory called `docs` and a `target` in your Makefile called `docs`, when you run `make docs` you'll get this message:

<div class="animated-code">

    ```console
    $ make docs
    make: `docs` is up to date.
    ```

</div>

We can fix this by defining a [`PHONY`](https://www.gnu.org/software/make/manual/make.html#Phony-Targets){:target="_blank"} target in our makefile by adding this line above the target:
```bash
.PHONY: <target_name>
```

Most of the rules in our Makefile will require the `PHONY` target because we want them to execute even if there is a file sharing the target's name. An exception to this is the `venv` target because we don't want to create a `venv` directory if it already exists.

## Prerequisites

Before we make a target, we can attach prerequisites to them. These can either be file targets that must exist or PHONY target commands that need to be executed prior to *making* this target. We use the *style* target as a prerequisite for the *clean* target so that all files are formatted appropriately prior to cleaning them.

```bash hl_lines="3"
# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage
```

## Variables
We can also set and use [variables](https://www.gnu.org/software/make/manual/make.html#Using-Variables){:target="_blank"} inside our Makefile to organize all of our rules.

- We can set the variables directly inside the Makefile. If the variable isn't defined in the Makefile, then it would default to any environment variable with the same name.
```bash
# Set variable
MESSAGE := "hello world"

# Use variable
greeting:
    @echo ${MESSAGE}
```

- We can also use variables passed in when executing the rule like so (ensure that the variable is not overriden inside the Makefile):
```bash
make greeting MESSAGE="hi"
```

## Shells

Each line in a recipe for a rule will execute in a separate sub-shell. However for certain recipes such as activating a virtual environment and loading packages, we want to execute all steps in one shell. To do this, we can add the [`.ONESHELL`](https://www.gnu.org/software/make/manual/make.html#One-Shell){:target="blank"} special target above any target.

```bash hl_lines="2"
# Environment
.ONESHELL:
venv:
    python3 -m venv ${name}
    source ${name}/bin/activate
    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install -e ".[dev]" --no-cache-dir
    pre-commit install
    pre-commit autoupdate
    pip uninstall dataclasses -y
```

However this is only available in Make version 3.82 and above and most Macs currently use version 3.81. You can either update to the current version or chain your commands with `&&`.

```bash
# Environment
venv:
    python3 -m venv ${name}
    source ${name}/bin/activate && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -e ".[dev]" --no-cache-dir && \
    pre-commit install && \
    pre-commit autoupdate && \
    pip uninstall dataclasses -y
```

> There's a whole lot [more](https://www.gnu.org/software/make/manual/make.html){:target="_blank"} to Makefiles but this is plenty for most applied ML projects.

<!-- Citation -->
{% include "cite.md" %}